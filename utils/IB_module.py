import torch.nn as nn
import torch
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from dataloaders.data_thinv import convert_one_hot_new
import torch.nn.functional as F


class TorchWelfordEstimator(nn.Module):  # 用于计算所有特征点组成的均值和方差1
    """
    Estimates the mean and standard derivation.
    For the algorithm see ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``.

    Example:
        Given a batch of images ``imgs`` with shape ``(10, 3, 64, 64)``, the mean and std could
        be estimated as follows::

            # exemplary data source: 5 batches of size 10, filled with random data
            batch_generator = (torch.randn(10, 3, 64, 64) for _ in range(5))

            estim = WelfordEstimator(3, 64, 64)
            for batch in batch_generator:
                estim(batch)

            # returns the estimated mean
            estim.mean()

            # returns the estimated std
            estim.std()

            # returns the number of samples, here 10
            estim.n_samples()

            # returns a mask with active neurons
            estim.active_neurons()
    """

    def __init__(self):
        super().__init__()
        self.device = None  # Defined on first forward pass
        self.shape = None  # Defined on first forward pass
        self.register_buffer('_n_samples', torch.tensor([0], dtype=torch.long))

    def _init(self, shape, device):
        self.device = device
        self.shape = shape
        self.register_buffer('m', torch.zeros(*shape))
        self.register_buffer('s', torch.zeros(*shape))
        self.register_buffer('_neuron_nonzero', torch.zeros(*shape, dtype=torch.long))
        self.to(device)

    def forward(self, x):
        """ Update estimates without altering x """
        if self.shape is None:
            # Initialize running mean and std on first datapoint,将标准差和均值全部初始化为0
            self._init(x.shape[1:], x.device)
        # 计算每batch样本的均值和标准差，保存在self.m 和self.s
        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_m = self.m.clone()
            # Update the mean: new_m = old_m + (x-m)/(n+1)
            self.m = self.m + (xi - self.m) / (self._n_samples.float() + 1)
            # Update the 无偏估计: s
            self.s = self.s + (xi - self.m) * (xi - old_m)
            self._n_samples += 1
        return x

    def n_samples(self):
        """ Returns the number of seen samples. """
        return int(self._n_samples.item())

    def mean(self):
        """ Returns the estimate of the mean. """
        return self.m

    def std(self):
        """returns the estimate of the standard derivation."""
        return torch.sqrt(self.s / (self._n_samples.float() - 1))

    def active_neurons(self, threshold=0.01):
        """
        Returns a mask of all active neurons.
        A neuron is considered active if ``n_nonzero / n_samples  > threshold``
        """
        return (self._neuron_nonzero.float() / self._n_samples.float()) > threshold


# def reparameterize_Bernoulli(p_i, tau, num_sample=10):
#     p_i_ = p_i.view(p_i.size(0), 1, -1)
#     p_i_ = p_i_.expand(p_i_.size(0), num_sample, p_i_.size(-1))  # Batch size, Feature size,num_samples
#     C_dist = RelaxedBernoulli(tau, logits=p_i_)     # 宽松伯努利分布，重参数化使得可梯度回传
#     V = C_dist.sample().mean(dim=1)     # m 通过采样生成
#     return V

class VibModel(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, mask_mode='hard_mask', lamb=0.6, indim=1024, beta=0.3,
                 b_threshold=0.9, compression_mode='relaxed_bernoulli',
                 init_mag=9, init_var=0.01, kl_mult=1, divide_w=False):
        super(VibModel, self).__init__()

        self.estimator = TorchWelfordEstimator()
        self.compression_mode = compression_mode
        self.embedding = nn.Sequential(nn.Conv2d(indim, indim, 1), nn.ReLU())
        self.embedding_b = nn.Conv2d(indim, indim, 1)
        self.lamb = lamb
        self.channel_compressing = nn.Sequential(nn.Conv2d(indim, 1, 1),
                                                 nn.ReLU())

        self.soft_prediction = nn.Sequential(
            nn.Conv2d(indim, indim, 1),
            nn.ReLU(),
            nn.Conv2d(indim, 1, 1),
            # nn.Sigmoid(),
        )

        self.hard_prediction = nn.Sequential(
            nn.Conv2d(indim, indim, 1),  # V
            nn.ReLU(),
            nn.Conv2d(indim, 2, 1),
            nn.Softmax(dim=1),
        )

        self.sigmoid = nn.Sigmoid()

        # self.instance_classifier = nn.Linear(size[2], 1)
        #
        # self.fc6 = nn.Sequential(
        #     nn.Linear(size[2], size[3]),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(size[3], 1),
        #     nn.Sigmoid()
        # )

        self.bernoulli_threshold = b_threshold
        self.beta = beta
        self.mask_mode = mask_mode
        self.relu = True

        # 初始化 _mean, _std, _active_neurons
        self._mean = None
        self._std = None
        self._active_neurons = None

    def reparameterize_Bernoulli(self, p_i, tau, num_sample=5):

        p_i_ = p_i.unsqueeze(dim=1)
        p_i_ = p_i_.unsqueeze(1).expand(-1, num_sample, -1, -1, -1)
        C_dist = RelaxedBernoulli(tau, logits=p_i_)  # 宽松伯努利分布，重参数化使得可梯度回传
        V = C_dist.sample().mean(dim=1)  # m 通过采样生成
        V = V.squeeze(dim=1)
        return V

    def _bottleneck_projection_gaussion(self, x, lamb):
        """ Selectively remove information from x by applying noise """

        if self._mean is None:
            self._mean = self.estimator.mean()

        if self._std is None:
            self._std = self.estimator.std()

        if self._active_neurons is None:
            self._active_neurons = self.estimator.active_neurons()

        # addiing noise into Tensor R
        eps = x.data.new(x.size()).normal_()  # 生成噪声
        # To get the noise with the same mean and standard deviation as R
        eps = self._std * eps + self._mean
        z = lamb * x + (1 - lamb) * eps

        # Sample new output values from p(z|x)
        z *= self._active_neurons
        # Clamp output, if input was post-relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        return z

    def _kl_div(self, x, lambda_, mean_r, std_r):  # 计算KL散度
        """Computes the KL Divergence between the noise (Q(Z)) and
           the noised activations P(Z|R)).
        """
        # 正常化操作 [1]
        r_norm = (x - mean_r) / std_r

        # 计算 Z'|R' 的均值和方差 [2,3]
        var_z = (1 - lambda_) ** 2
        var_z = torch.tensor(var_z, device=x.device)  # 将 var_z 转换为 Tensor

        mu_z = r_norm * lambda_

        log_var_z = torch.log(var_z)  # 计算 var_z 的对数

        # 计算 KL 散度
        # 参见 eq. 7: https://arxiv.org/pdf/1606.05908.pdf
        capacity = -0.5 * (1 + log_var_z - mu_z ** 2 - var_z)
        return capacity

    def forward(self, X, mask=None, phase='new_train'):
        return_z = []
        return_kl = 0
        return_dsl = 0

        T, _, _, _ = X.size()
        for t in range(0, T):
            x = self.embedding(X[t])
            if self.compression_mode == 'adding_noise':
                self.estimator(x)  # 更新估计器，而不返回 x
                _mean = self.estimator.mean()  # 获取均值
                _std = self.estimator.std()  # 获取标准差
                _active_neurons = self.estimator.active_neurons()  # 获取激活神经元掩码
                z = self._bottleneck_projection_gaussion(x, self.lamb)  # 计算 Z
                KL = self._kl_div(x, self.lamb, _mean, _std) * _active_neurons  # 计算 KL 散度

            elif self.compression_mode == 'relaxed_bernoulli':
                # pixel_pred = self.channel_compressing(x)  # 压缩特征图维数到1
                # sigmoid_logits = torch.sigmoid(pixel_pred)  # sigmoid(logit)
                # inst_logits = nn.functional.logsigmoid(pixel_pred)  # logit
                # z_mask = self.reparameterize_Bernoulli(p_i=inst_logits, tau=0.1)  # 使用logit生成分布

                # pixel_pred = self.embedding_b(x)
                # sigmoid_logits = torch.sigmoid(pixel_pred)  # P(m_n | x_n)
                # inst_logits = nn.functional.logsigmoid(pixel_pred)
                pixel_pred = self.channel_compressing(x)  # 压缩特征图维数到1
                sigmoid_logits = torch.sigmoid(pixel_pred)  # sigmoid(logit)
                inst_logits = nn.functional.logsigmoid(pixel_pred)  # logit
                z_mask = self.reparameterize_Bernoulli(p_i=inst_logits, tau=0.1)

                mse_loss = torch.nn.MSELoss()
                KL = mse_loss(sigmoid_logits, torch.ones_like(sigmoid_logits) * self.bernoulli_threshold)
                KL = KL * self.beta
                z_mask = (z_mask + sigmoid_logits) / (1.0 + self.bernoulli_threshold)
                # z_mask = z_mask.expand(x.size(0), x.size(1), x.size(2), x.size(3))
                z = x * z_mask
                return_z.append(z)
                # print('z :',z.size())
            # if self.mask_mode =='unsupervised':
            #     return z, KL
            # elif self.mask_mode =='soft_mask' or self.mask_mode =='hard_mask':
            #     out = self.soft_prediction(z)
            #     out = nn.Sigmoid(out)
            #     mse_loss = torch.nn.MSELoss()
            #     deep_supervision_loss = mse_loss(out, mask)
            #     return z, KL, deep_supervision_loss
            # else:
            #     raise ValueError('Wrong mask mode')

            else:
                raise ValueError('Unexpected compression mode')

            if t != 0 and t != T-1:
                continue

            if self.mask_mode == 'unsupervised' or phase=='test':  # 不使用下采样mask
                continue
            elif self.mask_mode == 'soft_mask' or self.mask_mode == 'hard_mask':  # soft mask是线性差值得到的mask， hard mask是最近邻得到的mask，都需要转化成float
                out = self.soft_prediction(z)
                out = self.sigmoid(out)
                mse_loss = torch.nn.MSELoss()
                current_mask = convert_one_hot_new(mask, 1)[0, t]
                current_mask = F.max_pool2d(current_mask.float().unsqueeze(0), kernel_size=16)
                current_mask = current_mask.squeeze(0)
                # print('out: ',out.size())
                # print('mask: ', current_mask.size())
                deep_supervision_loss = mse_loss(out, current_mask)

                return_kl += KL
                return_dsl += deep_supervision_loss

        if self.mask_mode == 'unsupervised' or phase=='test':  # 不使用下采样mask
            return_z = torch.stack(return_z, dim=0)
            return return_z, return_kl
        elif self.mask_mode == 'soft_mask' or self.mask_mode == 'hard_mask':  # soft mask是线性差值得到的mask， hard mask是最近邻得到的mask，都需要转化成float
            return_z = torch.stack(return_z, dim=0)
            return return_z, return_kl, return_dsl  # KL和DSloss前分别乘一个系数，计算综合损失
        else:
            raise ValueError('Wrong mask mode')


#   测试：随机生成一个2*64*64*64的tensor，观察输出，z的维度和输入一致，KL和DS是scalar
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 32
    x = torch.Tensor(1024, image_size, image_size)
    mask = torch.Tensor(1, image_size, image_size)
    x.to(device)
    mask.to(device)
    print("x size: {}".format(x.size()))

    model = VibModel(mask_mode='hard_mask')

    # out1 = model(x)
    # out1, out2 = model(x)
    out1, out2, out3 = model(x, mask)
    print("out size: {}".format(out1.size()))
    print(out2)
    # print("out size: {}".format(out3.size()))
    print(out3)
    num_para = sum(p.numel() for p in model.parameters())
    print(num_para)
