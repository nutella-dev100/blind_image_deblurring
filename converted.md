# Blind Image Deblurring via a Novel Sparse Channel Prior 

![](https://cdn.mathpix.com/cropped/a5fb9c8e-dfb4-4ced-9129-e477b30585d0-1.jpg?height=59&width=903&top_left_y=511&top_left_x=116)<br>1 School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi 214122, China; yin_hefeng@jiangnan.edu.cn<br>${ }^{2}$ Jiangsu Provincial Engineering Laboratory of Pattern Recognition and Computational Intelligence, Jiangnan University, Wuxi 214122, China<br>* Correspondence: 7151905005@vip.jiangnan.edu.cn (D.Y.); wu_xiaojun@jiangnan.edu.cn (X.W.)

Citation: Yang, D.; Wu, X.; Yin, H. Blind Image Deblurring via a Novel Sparse Channel Prior. Mathematics 2022, 10, 1238. https://doi.org/ 10.3390/math10081238

Academic Editors: Jianping Gou, Weihua Ou, Shaoning Zeng and Lan Du

Received: 25 February 2022
Accepted: 6 April 2022
Published: 9 April 2022
Publisher's Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Copyright: © 2022 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https:// creativecommons.org/licenses/by/ 4.0/).


#### Abstract

Blind image deblurring (BID) is a long-standing challenging problem in low-level image processing. To achieve visually pleasing results, it is of utmost importance to select good image priors. In this work, we develop the ratio of the dark channel prior (DCP) to the bright channel prior (BCP) as an image prior for solving the BID problem. Specifically, the above two channel priors obtained from RGB images are used to construct an innovative sparse channel prior at first, and then the learned prior is incorporated into the BID tasks. The proposed sparse channel prior enhances the sparsity of the DCP. At the same time, it also shows the inverse relationship between the DCP and BCP. We employ the auxiliary variable technique to integrate the proposed sparse prior information into the iterative restoration procedure. Extensive experiments on real and synthetic blurry sets show that the proposed algorithm is efficient and competitive compared with the state-of-the-art methods and that the proposed sparse channel prior for blind deblurring is effective.


Keywords: blind image deblurring; image prior; sparse channel; sparsity
MSC: 68U10

## 1. Introduction

The goal of blind image deblurring is to restore a sharp image and a blur kernel from the input degraded image. The degradation types include motion blur, noise, out-of-focus and camera shake. Assuming that the blur is uniform and spatially invariant, the mathematical formulation of the blurring process can be modeled as

$$
\begin{equation*}
b=l * k+n \tag{1}
\end{equation*}
$$

where $b$ is the blurry input, $k$ is the blur kernel and $n$ is the additive noise. The $*$ denotes the convolution operator. This problem is highly ill-posed because both the latent sharp image $l$ and blur kernel $k$ are unknown. In order to make this problem well-posed, most existing methods utilize the statistics of natural images to estimate the blur kernel. For example, a heavy-tailed distribution [1], patch recurrence prior [2], nuclear norm [3,4], low-rank prior [5], sparse prior [6], multiscale latent prior [7] or additional information of a specific image [8-10] have been used to estimate a better kernel.

Strong sparsity of image intensity and gradient has been widely used in low-level computer vision processing problems. It also has mature applications in the field of image deblurring [6,11-13], such as the $L_{1} / L_{2}$ [14] norm, the reweighted $L_{1}$ norm [15], the $L_{0}$ norm prior [16-19] and the sparse prior-local maximum gradient (LMG) [20]. For favoring clear images over blurry ones, the edge selection method [21-23] is embedded in the blind deconvolution framework. However, strong edges are not always available in many cases. The channel prior was introduced by He et al. for image defogging in Ref. [24]. Then, Pan et al. [18] enforced the sparsity of the dark channel by the $L_{0}$ norm for kernel estimation. Unfortunately, this prior does not work well on images with large noise and large numbers
of pixels. To solve this problem, Yan et al. [19] proposed an extreme channel prior (ECP) which utilizes both the dark channel and bright channel for estimating the blur kernel.

In this paper, a novel sparse channel prior is proposed for blind image deblurring. Inspired by [18,19,24], we take the advantages of the DCP and BCP to construct a confrontation constraint $\mathrm{D} / \mathrm{B}$. We prove its characteristic from a mathematical perspective and explore how these properties can be used to estimate the blur kernel. In the proposed algorithm, the optimization of the proposed prior is a challenging problem. We use the idea of auxiliary variables and the alternating minimization method to decompose the problem into independent subproblems optimised by the alternating direction minimization (ADM) method. The main contributions of this work can be stated as follows:

- A new D/B prior is presented for kernel estimation, which fully explores the relationship between the DCP and BCP. We also verify the effectiveness of D/B.
- We develop an effective optimization strategy for kernel estimation based on the idea of auxiliary variables and the alternating direction minimization (ADM) method.
- Experiments on four databases show that the proposed method is competitive compared with the state-of-the-art blind deblurring algorithms.

The rest of this paper is organized as follows. Section 2 introduces the related work. The proposed D/B is detailed in Section 3. Our blind deblurring model and optimization strategy are presented in Section 4. Section 5 shows the experimental results. Further discussion of our proposed deblurring algorithm is given in Section 6. Section 7 summarizes this paper.

## 2. Related Work

Blind image deblurring algorithms have made great progress due to the use of the proper kernel estimation model. In this part, we introduce the methods related to our work in an appropriate context.

The success of many blind image deblurring algorithms is based on the use of the statistical characteristics of the image intensity and gradient. Krishnan et al. [14] presented the $L_{1} / L_{2}$ norm based on the sparsity of image intensity. The $L_{1} / L_{2}$ norm is a normalized version of $L_{1}$, which enhances the sparsity of $L_{1}$. Levin et al. [1] observed the heavy-tailed distribution of image intensities and introduced a maximum posteriori (MAP) framework. Shan et al. [25] introduced a probability model to fit the sparse gradient distribution of a natural image. Pan et al. [16] developed a method in which both intensity and gradient are regularized by the $L_{0}$ norm for text image deblurring. These methods are limited by the modeling of more complex image structures and contexts.

Another group of blind image deblurring methods [22,23] employs a significant edge detection step for kernel estimation. Specifically, Cho et al. [21] predicted sharp edges by the bilateral and shock filters. Joshi et al. [26] detected image contours by locating the subpixels' extrema. These methods cannot capture the sparse kernel and structures, which makes the restored image blurry and noisy sometimes. To solve these problems, researchers have proposed many better models to estimate the blur kernel. Xu et al. [27] presented a two-phase kernel estimation algorithm, which separates kernel initialization from the iterative support detection (ISD)-based kernel refinement step, giving an efficient estimation process and maintaining many small structures. Zoran and Weiss [28] proposed the expected patch log likelihood (EPLL) method, which imposes a prior on the patches of the final image. However, this will iteratively restore the degradation. Vardan et al. [29] exploited the multiscale prior to further improve the EPLL and reduce the error to that of the global modeling. Bai et al. [7] developed a multiscale latent structures (MSLS) prior. Based on the MSLS prior, their deblurring algorithm consists of two stages: sharp image estimation in the coarse scales and a refinement process in the finest scale. For the patch-based methods, global modeling is a difficult problem.

With the rapid development of the deep learning method, remarkable results have been achieved in the field of blind image deblurring [30-34]. For example, convolutional neural networsk (CNN) [35], Wasserste generative adversarial networks (GAN) [36], deep hierarchical multipatch networks (DMPHN) [37], ConvLSTM [38] and scale-recurrent networks (SRN) [39] are all designed for image deblurring. Zheng et al. [40] presented an edge heuristic multiscale GAN, which utilizes the edge's information to conduct the deblurring process in a coarse-to-fine manner for nonuniform blur. Liang et al. [41] learned novel neural network structures from RAW images and achieved superb performance. Chang et al. [42] proposed a long-short-exposure fusion network (LSFNet) for low-light image restoration by using the pairs of long- and short-exposure images. The success of deep-learning-based methods mainly relies on the consistency between training and test data, which limits the generalization ability of these methods.

Recently, the classical dark channel prior (DCP) has been proved effective for image deblurring. The DCP was introduced by He et al. [24] for image defogging. It is based on the observation that there is at least one color channel that has very low and close-to-zero pixel values on outdoor haze-free nonsky image patches. Pan et al. [18] further found that most elements of the dark channel are zero for nature images and then enhanced the sparsity of dark channel for image deblurring. Inspired by the DCP, the bright channel prior (BCP) is proposed. That is, in most of nature patches, at least one color channel has very high pixel values. Yan et al. [19] used the simple addition of the DCP and BCP to form an extreme channel prior (ECP) for a blind image deblurring algorithm. However, the relationship between the BCP and DCP is not fully explored in the ECP.

## 1. Proposed Sparse Channel Prior

To explain that the proposed sparse channel vary after blurring, we model the blurring process as described in [43]. For an image $I$, consider the noise is small enough to be neglected. We have:

$$
\begin{equation*}
b(x)=\sum_{z \in \Psi(x)} l\left(x+\left[\frac{m}{2}\right]-z\right) k(z) \tag{2}
\end{equation*}
$$

where $x$ and $m$ denote the coordinates of the pixel and the size of the blur kernel $k$, respectively. $\Psi(x)$ represents an image patch centered at $x, \sum_{z \in \Psi(x)} k(z)=1$ and $k(z) \geq 0$. [.] is a rounding operator.

Inspired by the two channels (dark and bright channels) and the statistics of images, we observe that when the dark channel is more different from the bright channel of one image patch, the edges are more salient, which is helpful to estimate an accurate blur kernel. To formally describe this observation, the proposed sparse channel prior is defined by:

$$
\begin{align*}
R(x) & =\min _{y \in \Psi(x)}\left(\min _{c \in(r, g, b)}\left(I^{c}(y)\right)\right) \\
& /\left(\max _{y \in \Psi(x)}\left(\max _{c \in(r, g, b)}\left(I^{c}(y)\right)\right)+\epsilon\right)  \tag{3}\\
& =D(x) /(B(x)+\epsilon)
\end{align*}
$$

where $x$ and $y$ denote the coordinates of the pixel, $\epsilon$ is a non-negative constant and $\Psi(x)$ represents an image patch centered at $x . I^{c}$ is the $c$-th color channel of image $I$. As described in Equation (3), $\mathrm{B}(x)=\max _{y \in \Psi(x)}\left(\max _{c \in(r, g, b)}\left(I^{c}(y)\right)\right)$ represents the BCP and $D(x)=\min _{y \in \Psi(x)}\left(\min _{c \in(r, g, b)}\left(I^{c}(y)\right)\right)$ represents the DCP. Dark channels are obtained by two minimization operations: $\min _{c \in(r, g, b)}$ and $\min _{y \in \Psi(x)}$. The bright channel is obtained by two maximization operations: $\max _{c \in(r, g, b)}$ and $\max _{y \in \Psi(x)}$. In the implementations of the DCP and BCP, if $I$ is a gray image, then only the latter operation is performed. A small value of $R(x)$ implies there are salient edges in the image patch. On the contrary, a large $R(x)$ implies that there are fine structures in an image patch. The reason is that when the edge is salient, the pixel values are more different between the two sides of edges. It means that the minimum value is more different from the maximum value of the image patch. Conversely, when the difference between the DCP and BCP is not that large, the image edge is unclear, and the value of $R(x)$ is large. Therefore, it is natural to think that if the

DCP is equal to or slightly smaller than the BCP, small edges can be accurately removed by minimizing Equation (3).

Consider a natural image that was blurred by a blur kernel. Blur reduces the maximum pixel value and increases the minimum pixel value of one patch. In other words, the DCP of one patch will increase and the BCP will decrease. Let $R(b)$ and $R(l)$ denote the proposed sparse channel of the blurred and clear image, respectively, when the $l(x)= \max _{y \in \Psi(x)} l(y)=\min _{y \in \Psi(x)} l(y), R(b)(x) \geq R(l)(x)$. To further apply this proposition to the definition of the proposed sparse channel, we have:

$$
\begin{align*}
R(b)(x) & =\frac{\min _{y \in \Psi(x)}\left(\min _{c \in(r, g, b)}\left(b^{c}(y)\right)\right)}{\max _{y \in \Psi(x)}\left(\max _{c \in(r, g, b)}\left(b^{c}(y)\right)\right)+\epsilon} \\
& =\frac{\min _{y \in \Psi(x)} b(y)}{\max _{y \in \Psi(x)} b(y)+\epsilon} \\
& =\frac{\min _{y \in \Psi(x)} \sum_{z \in \Phi(x)} l\left(y+\left[\frac{m}{2}\right]-z\right) k(z)}{\max _{y \in \Psi(x)} \sum_{z \in \Phi(x)} l\left(y+\left[\frac{m}{2}\right]-z\right) k(z)+\epsilon} \\
& \geq \frac{\sum_{z \in \Phi(x)} \min _{y \in \Psi(x)} l\left(y+\left[\frac{m}{2}\right]-z\right) k(z)}{\sum_{z \in \Phi(x)} \max _{y \in \Psi(x)} l\left(y+\left[\frac{m}{2}\right]-z\right) k(z)+\epsilon}  \tag{4}\\
& \geq \frac{\sum_{z \in \Phi(x)} \min _{\widehat{y} \in \widehat{\Psi}(x)} l\left(\widehat{y}+\left[\frac{m}{2}\right]-z\right) k(z)}{\sum_{z \in \Phi(x)} \max _{\widehat{y} \in \widehat{\Psi}(x)} l\left(\widehat{y}+\left[\frac{m}{2}\right]-z\right) k(z)+\epsilon} \\
& =\frac{\min _{\widehat{y} \in \widehat{\Psi}(x)} l(\widehat{y})}{\max _{\widehat{y} \in \widehat{\Psi}(x)} l(\widehat{y})+\epsilon} \\
& =R(l)(x)
\end{align*}
$$

Let $\widehat{m}$ and $S_{\Psi}$ denote the size of $\widehat{\Psi}(x)$ and $\Psi(x)$, respectively. Then we have $\widehat{m}=S_{\Psi}+m$. Equation (4) shows that $R(x)$ of the image patch centered at $x$ after blurring is no less than the value of the original image patch centered at $x$.

Equation (4) proves $R(l)(x) \leq R(b)(x)$. This means that after blurring, the difference between the DCP and the BCP is smaller than that of the corresponding patch in a sharp image. In other words, $R(x)$ always favors the sharp image. We further validate our analysis on the dataset [44]. Figure 1a-c show the histogram of the average number of dark channel pixels, bright channel pixels and D/B channel pixels, respectively. As can be observed, a large portion of the pixels in the dark channels and bright channels possess very small or large values, and our D/B channel pixels possess smaller values than those of the DCP and BCP. As shown in Figure 1, the proposed sparse channels of clear images have significantly more zero elements than those of blurred images. Thus, the sparsity of the proposed channel is a natural metric to distinguish clear images from blurred images. This observation motivates us to introduce a new regularization term to enforce sparsity of the proposed channels in latent images.

## 2. Proposed Sparse Channel as an Image Prior

Equation (4) shows that after blurring, the difference between the DCP and BCP is smaller than that of the corresponding patch in a sharp image. Therefore, in order to generate sharp and reliable salient edges, we propose a novel sparse channel prior which combines the D/B and $L_{0}$ norm:

$$
\begin{equation*}
P(x)=\frac{\|D(x)\|_{0}}{\|B(x)\|_{0}+\epsilon} \tag{5}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/160379be-c237-4fc5-98ae-8898d19f11b6-3.jpg?height=546&width=1732&top_left_y=333&top_left_x=163)
Figure 1. The statistics of the DCP, the BCP and our proposed D/B prior: (a-c) average channel pixels distribution of bright, dark and our D/B, respectively.

We define $P(x)$ as a $\mathrm{D} / \mathrm{B}$ prior, and the $L_{0}$ norm is used for sparsity. Let $\Psi(x)$ denote one patch of the image $I$. If there exist some pixels $x \in \Psi(x)$ such that $I(x)=0$, we have

$$
\begin{equation*}
P(b)(x) \geq P(l)(x) \tag{6}
\end{equation*}
$$

where $P(b)(x)$ and $P(l)(x)$ denote the $\mathrm{D} / \mathrm{B}$ prior of the blurred and clear image, respectively. This property directly follows from Equation (4). In the framework of MAP, by minimizing the sparse prior $P(x)$, we obtain a result that favors a sharp image. This property is also validated using dataset [44]. As shown in Figure 1c, the average number of D/B channels in clear images has significantly more zero elements than that of blurred ones.

## 3. Proposed Blind Deblurring Model

Based on the proposed D/B prior, we construct the blind deblurring model under the maximum a posteriori (MAP) framework.

$$
\begin{equation*}
\operatorname{argmin}_{l, k}\|l \otimes k-b\|_{2}^{2}+\mu P(l)+\vartheta\|\nabla l\|_{0}+\gamma\|k\|_{2}^{2} \tag{7}
\end{equation*}
$$

where $P(l)$ is our proposed prior, ∇ denotes the gradient operation and $\mu, \vartheta$ and $\gamma$ are non-negative weights. The data-fitting term of our model ensures that the latent sharp image is consistent with the observed image. $\|\nabla l\|_{0}$ is the $L_{0}$ norm of the image gradient, which is used to suppress ringing and artifacts. Finally, we use the $L_{2}$ norm to increase the sparsity of the blur kernel.

### 3.1. Optimization

In this part, we adopt the ADM method to obtain the solution to the objective function. By using the idea of alternating optimization, we can obtain two independent subproblems about $l$ and $k$, respectively:

$$
\begin{equation*}
\operatorname{argmin}_{l}\|l \otimes k-b\|_{2}^{2}+\mu \frac{\|D(l)\|_{0}}{\|B(l)\|_{0}+\epsilon}+\vartheta\|\nabla l\|_{0} \tag{8}
\end{equation*}
$$

and

$$
\begin{equation*}
\operatorname{argmin}_{k}\|l \otimes k-b\|_{2}^{2}+\gamma\|k\|_{2}^{2} \tag{9}
\end{equation*}
$$

Equation (9) is a classical least squares problem with respect to $k$. By introducing the auxiliary variable $g$, which is related to $\nabla l$, Equation (8) can be written as follows:

$$
\begin{equation*}
\operatorname{argmin}_{l, g}\|l \otimes k-b\|_{2}^{2}+\lambda\|\nabla l-g\|_{2}^{2}+\mu \frac{\|D(l)\|_{0}}{\|B(l)\|_{0}+\epsilon}+\vartheta\|g\|_{0} \tag{10}
\end{equation*}
$$

Equation (10) can be decomposed into:

$$
\begin{equation*}
\operatorname{argmin}_{l}\|l \otimes k-b\|_{2}^{2}+\lambda\|\nabla l-g\|_{2}^{2}+\mu \frac{\|D(l)\|_{0}}{\|B(l)\|_{0}+\epsilon} \tag{11}
\end{equation*}
$$

and

$$
\begin{equation*}
\operatorname{argmin}_{g} \lambda\|\nabla l-g\|_{2}^{2}+\vartheta\|g\|_{0} \tag{12}
\end{equation*}
$$

Equation (12) is an $L_{0}$ norm minimization problem for $g$.

### 3.2. Estimating Intermediate Image l

For the $k$-th iteration, we consider $B(l)$ estimated in the $(k-1)$-th iteration as a constant. Denoting

$$
\begin{equation*}
w_{k}=\mu /\left(\|B(l)\|_{0}+\epsilon\right) \tag{13}
\end{equation*}
$$

Equation (11) can be rewritten as follows:

$$
\begin{equation*}
\operatorname{argmin}_{l}\|l \otimes k-b\|_{2}^{2}+\lambda\|\nabla l-g\|_{2}^{2}+w_{k}\|D(l)\|_{0} \tag{14}
\end{equation*}
$$

By introducing an auxiliary variable, $p$, which is related to $D(l)$, Equation (14) can be reformulated as follows:

$$
\begin{equation*}
\operatorname{argmin}_{l, p}\|l \otimes k-b\|_{2}^{2}+\xi\|D(l)-p\|_{2}^{2}+\lambda\|\nabla l-g\|_{2}^{2}+w_{k}\|p\|_{0} \tag{15}
\end{equation*}
$$

Using the idea of alternating optimization, we can obtain two independent subproblems to solve for $l$ and $p$, respectively:

$$
\begin{equation*}
\operatorname{argmin}_{l}\|l \otimes k-b\|_{2}^{2}+\xi\|D(l)-p\|_{2}^{2}+\lambda\|\nabla l-g\|_{2}^{2} \tag{16}
\end{equation*}
$$

and

$$
\begin{equation*}
\operatorname{argmin}_{p} \xi\|D(l)-p\|_{2}^{2}+w_{k}\|p\|_{0} \tag{17}
\end{equation*}
$$

Equation (16) contains all quadratic terms, and we can obtain its solution by the least squares method. In each iteration, the FFT (Fast Fourier Transform) is used to accelerate the computation process. Its closed-form solution is given as follows:

$$
\begin{equation*}
l=\mathcal{F}^{-1}\left(\frac{\overline{\mathcal{F}(k)} \mathcal{F}(b)+\xi \mathcal{F}(p)+\lambda \mathcal{F}_{g}}{\overline{\mathcal{F}(k)} \mathcal{F}(k)+\lambda \overline{\mathcal{F}(\nabla)} \mathcal{F}(\nabla)+\xi}\right) \tag{18}
\end{equation*}
$$

where $\mathcal{F}_{g}=\left(\overline{\mathcal{F}\left(\nabla_{v}\right)} \mathcal{F}\left(g_{v}\right)+\overline{\mathcal{F}\left(\nabla_{\mathrm{h}}\right)} \mathcal{F}\left(g_{\mathrm{h}}\right)\right)$ and $\mathcal{F}(\cdot)$ and $\mathcal{F}^{-1}(\cdot)$ are the Fast Fourier Transform (FFT) and its inverse, respectively. $\overline{\mathcal{F}(\cdot)}$ denotes the complex conjugate operator of FFT and $\nabla_{v}$ and $\nabla_{h}$ are gradients in the vertical and horizontal directions, respectively.

### 3.3. Estimating $p$ and $g$

Equations (12) and (17) are minimization problems of the $L_{0}$ norm. Due to the difficulty of solving the $L_{0}$ norm minimization problem, we adopt the method described in Ref. [13]. As a result, the solution of Equation (17) can be expressed as:

$$
p=\left\{\begin{array}{cc}
D(l), & D(l) \geq \frac{w_{k}}{\xi}  \tag{19}\\
0, & \text { otherwise }
\end{array}\right.
$$

Given $l$, the solution of Equation (12) can be expressed as:

$$
g= \begin{cases}\nabla l, & |\nabla l|^{2} \geq \frac{\vartheta}{\lambda}  \tag{20}\\ 0, & \text { otherwise }\end{cases}
$$

### 3.4. Estimating Blur Kernel $k$

Since the updating of the blur kernel is an independent subproblem, we estimate $k$ in the gradient space. Specifically, we obtain the solution to the blur kernel by minimizing the following problem though the known intermediate image $l$ :

$$
\begin{equation*}
\min _{k}\|\nabla l \otimes k-\nabla y\|_{2}^{2}+\gamma\|k\|_{2}^{2} \tag{21}
\end{equation*}
$$

where ∇ denotes the gradient operation. Note that we use Equation (21) to estimate the blur kernel instead of Equation (9), which helps to suppress ringing artifacts and eliminate noise. The closed-form solution to Equation (21) is obtained by FFT.

$$
\begin{equation*}
k=\mathcal{F}^{-1}\left(\frac{\overline{\mathcal{F}(\nabla l)} \mathcal{F}(\nabla y)}{\overline{\mathcal{F}(\nabla l)} \mathcal{F}(\nabla l)+\gamma}\right) \tag{22}
\end{equation*}
$$

The coarse-to-fine strategy is used in the process of blur kernel estimation, which is similar to that used in [26,45]. In the process of solving the problem, it is very important to restrict the small values of the blur kernel by thresholding at fine scale, which enhances the robustness of the algorithm to noise.

### 3.5. Estimating Latent Sharp Image

Although the latent sharp images can be estimated from Equation (18), this formulation is less effective for fine-texture details. For the purpose of suppressing ringing and artifacts, we fine-tune the final restored image. With the estimated blur kernel and blur input image $y$, we can use the nonblind deconvolution method to obtain the final latent sharp image $l_{\text {latent }}$. Algorithm 1 summarizes the main steps of the final latent sharp image restoration method. Firstly, we estimate the restored image $l_{h}$ by the method in Ref. [46] using the hyper-Laplacian prior. Then we restore image $l_{r}$ according to the method in Ref. [47] using the total variation prior. Finally, the latent sharp image $l_{\text {latent }}$ is calculated by the average of the two restored images, i.e., $l_{\text {latent }}=\left(l_{h}+l_{r}\right) / 2$. The main steps of our proposed algorithm are summarized as Algorithm 2.

```
Algorithm 1 Final latent sharp image restoration.
Input: Blurry image $b$ and estimated kernel $k$.
    Estimate latent image $l_{h}$ by using the method described in [46] with Laplacian prior;
    Estimate latent image $l_{r}$ by using the method described in [47] with total variation prior;
    Restore the final sharp image $l_{\text {latent }}$ :
    $l_{\text {latent }}=\left(l_{h}+l_{r}\right) / 2$.
Output: Sharp latent image $l_{\text {latent }}$.
```

```
Algorithm 2 The proposed blind deblurring algorithm.
Input: Blurry image $y$;
        Initialize the intermediate image $l$ and blur kernel $k$;
        Estimate blur kernel $k$ from $b$;
        Alternately calculate $l$ and $k$ by the manner of coarse-to-fine levels:
            Estimate intermediate image $l$ by Equation (18);
            Estimate blur kernel $k$ by Equation (22);
        Interpolate solution to finer level as initialization;
        Calculate the latent sharp image according to Algorithm 1.
Output: Sharp latent image $l_{\text {latent }}$.
```

We first initialize the intermediate image $l$ and blur kernel $k$ according to the blurry input. Then we alternately update $l$ and $k$. In order to avoid falling into a local minimum, our algorithm is executed in a coarse-to-fine manner. The results of the coarse layer are

