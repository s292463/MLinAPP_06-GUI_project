
x
volta_cgemm_64x32_tn^?h*?2?8?ܘ@???H???Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  HB
J
redzone_checker*?2?@8???@??Hۑbmodel/Conv1D_4/conv1dh$u  ?B
J
redzone_checker*?2?@8???@ۯHۚbmodel/Conv1D_3/conv1dh u  ?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)v?R* 2?8???@??ZH??[bmodel/Conv1D_2/conv1dhu  HB
J
redzone_checker*?2?@8???@??H??bmodel/Conv1D_1/conv1dhu  ?B
n
redzone_checker*?2?@8???@??HۏXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2y8???@Ʒ>H??>Xb8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterhu  ?B
J
redzone_checker*?2?@8⽰@??H??bmodel/Conv1D_2/conv1dhu  ?B
n
redzone_checker*?2?@8???@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
o
redzone_checker*?2?@8???@??HۏXb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
o
redzone_checker*?2?@8???@??H??Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  ?B
R
volta_cgemm_64x32_tn^?h*?2?8???@??vH??vbmodel/Conv1D_3/conv1dhu  HB
w
volta_cgemm_64x32_tn^?h*?2?8???@??qH??qXb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  HB
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)8?A*?2?B8???@??pH??pXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
n
redzone_checker*?2?@8???@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)8?A*?2!?8й?@??`H??`Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
o
redzone_checker*?2?@8???@??H??Xb8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterhu  ?B
o
redzone_checker*?2?@8???@??H??Xb8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterhu  ?B
?
?void explicit_convolve_sgemm<float, int, 512, 6, 8, 3, 3, 5, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*)y?R* 2?8???@ܫVH??Wbmodel/Conv1D_2/conv1dhu  HB
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 6, 7, 3, 3, 5, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)B?2* 28???@??VH??VXb8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)G ?B*?2`y8?@??JH??LXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)8?A*?2{B8ᅖ@??JH??KXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)D?2* 28???@??DHĽDXb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)@ ?B*?2?M8??|@ǿ;H??AXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)S?*2y8ҍm@??6H??6Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  zB
?
?void wgrad_alg0_engine<float, 128, 6, 7, 3, 3, 5, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)K?2* 2y8??h@??H׏Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  ?B
?
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)G ?B*?2`R8??h@??4H??4Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void implicit_convolve_sgemm<float, float, 512, 6, 8, 3, 3, 5, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)v?R* 258??`@??H֚bmodel/Conv1D_3/conv1dhu  HB
?
?void cudnn::detail::dgrad_alg1_engine<float, 512, 6, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)^?*2?8??U@??*H??*Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  zB
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, true>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)P?*28??D@??"H??"Xb8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterhu  ?B
R
volta_gcgemm_64x32_nt~??*?28??=@??H??bmodel/Conv1D_2/conv1dhu  B
v
volta_gcgemm_64x32_nt~??*?28??4@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  B
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)@ ??*?2?8??2@??H??bmodel/Conv1D_3/conv1dhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??.@??H??Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??.@??H??bmodel/Conv1D_3/conv1dhu  HB
?
?void wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)P?*2y8??-@??	H??	Xb8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterhu  ?B
?
?void explicit_convolve_sgemm<float, int, 512, 6, 8, 3, 3, 5, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*)y?R* 258̮-@??H??bmodel/Conv1D_3/conv1dhu  HB
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*)+*?28??*@??H??bmodel/Conv1D_2/conv1dhu  ?B
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??(@??H׼Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  HB
?
?void fft1d_c2r_256<float2, float, float, false, true, false, false>(float*, float2 const*, int3, int3, int2, int, float, float, float*, float*)J??*2?8??(@??H??bmodel/Conv1D_1/conv1dhu  B
?
)volta_scudnn_128x128_stridedB_small_nn_v1???*?28??!@??
H??Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  HB
?
?void fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)@ ??*?2?8??!@??H??Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  HB
?
?void fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)@ ??*?2?8??@??H??bmodel/Conv1D_3/conv1dhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??bmodel/Conv1D_2/conv1dhu ?;B
u
volta_cgemm_64x32_tn^?h*?2?8??@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  HB
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu ?;B
?
?void wgrad_alg0_engine<float, 512, 6, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)^?*2	y8??@??H??Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  zB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2?8??@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu ?;B
?
?void cudnn::cnn::dgrad2d_grouped_direct_kernel<float, float, float, false, true, 0, 0>(cudnn::cnn::GroupedDirectParams, float const*, float const*, float*, float, float),*?2P8֫@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
?void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)S?*2y8??@??H??Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  zB
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2?8??@??H??bmodel/Conv1D_2/conv1dhu ?;B
u
volta_cgemm_32x32_tnb?H*@2?8??@??H??Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  /B
?
?void explicit_convolve_sgemm<float, int, 128, 5, 5, 3, 3, 3, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*)P?*2?8??@ېH??bmodel/Conv1D_1/conv1dhu  ?B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*2?8ע@ێH??bmodel/Conv1D_1/conv1dhu  ?B
P
volta_cgemm_32x32_tnb?H*@2?8ׄ@??
H??
bmodel/Conv1D_4/conv1dhu  /B
?
?void gemmk1_kernel<float2, 256, 5, false, false, true, false, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>(cublasGemmk1Params<float2, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2, biasType<cublasGemvTensorStridedBatched<float2>::value_type, float2>::type>)?*?2?8??@??H??bmodel/Conv1D_1/conv1dhu  ?B
d
'volta_scudnn_128x32_relu_interior_nn_v1??**@2?8??@??H??bmodel/Conv1D_1/conv1dhu  HB
e
'volta_scudnn_128x64_relu_interior_nn_v1???*?28??@??	H??	bmodel/Conv1D_4/conv1dhu  HB
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?2`8??@ܰH??Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*). ?D*?2?8??@??H݈Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 0, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?2y]8??@??H??b6gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGradhu  ?B
v
volta_gcgemm_64x32_nt~??*?28??@??H??Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  B
?
?void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)P?*28??@??H??bmodel/Conv1D_4/conv1dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?2y8??@??H??bmodel/Conv1D_4/conv1dhu  ?B
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*)+*?28??@??H??bmodel/Conv1D_3/conv1dhu  ?B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 0, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?2y`8??@??H??b6gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGradhu  ?B
R
volta_gcgemm_64x32_nt~??*?28??@??H??bmodel/Conv1D_3/conv1dhu  B
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*)+*?28??@??H??bmodel/Conv1D_4/conv1dhu  ?B
?
?void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 0, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor) ?*?2yR8??@??H??b6gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??	@??	H??	b%gradient_tape/model/Conv1D_1/ReluGradhu  ?B
?
?void cudnn::detail::dgrad_alg1_engine<float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, float, int)H?*28??	@??H߉Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b=model/Conv1D_1/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)H?*28ݼ@߼H??Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
?
?void explicit_convolve_sgemm<float, int, 1024, 5, 5, 3, 3, 3, 0, false>(int, int, int, float const*, int, float const*, int, float*, kernel_conv_params, unsigned long long, int, unsigned long long, int, float, float, int, float const*, float const*)??*28ݰ@??H??bmodel/Conv1D_4/conv1dhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b^gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?2(8??@??H??bmodel/Conv1D_1/BiasAddhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b^gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilter-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
evoid fft1d_r2c_256<float, float, float2, false, false>(float2*, float const*, int3, int3, int2, int2)B??*28??@??H߀bmodel/Conv1D_1/conv1dhu  B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu ?;B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?2(8ݯ@ݯHݯbmodel/Conv1D_1/Reluhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?2(8??@??H??b0gradient_tape/model/Conv1D_1/BiasAdd/BiasAddGradhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?28??@??HߙXb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2?.8??@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  ?B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*). ?D*?2?8??@??H??bmodel/Conv1D_4/conv1dhu  ?B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?2y8??@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??bmodel/Conv1D_3/conv1dhu ?;B
?
?void fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*). ?D*?2?8??@߿H??Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)9 ?*?2y8??@??H??bmodel/MaxPooling1D_1/MaxPoolhu??B
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?2`8??@ߵH??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??bmodel/Conv1D_3/conv1dhu ?;B
Q
volta_gcgemm_32x32_nt??`*@28??@??Hߟbmodel/Conv1D_4/conv1dhu  ?A
?
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)- ?L*?28??@??Hߝbmodel/Conv1D_4/conv1dhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  2 8??@??H??b0gradient_tape/model/Conv1D_1/BiasAdd/BiasAddGradhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2y`8ލ@??H??Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  /B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H߯bmodel/Conv1D_2/conv1dhu ?;B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??b%gradient_tape/model/Conv1D_2/ReluGradhu  ?B
v
volta_gcgemm_64x32_nt~??*?28??@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??bDmodel/MaxPooling1D_1/MaxPool-0-2-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b^gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu ?;B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b\gradient_tape/model/MaxPooling1D_1/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b_gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInput-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8޿@޿H޿b=model/Conv1D_2/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b^gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?2(8??@??H??bmodel/Conv1D_2/BiasAddhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??bmodel/Conv1D_4/conv1dhu ?;B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?2(8??@??H??b0gradient_tape/model/Conv1D_2/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8??@??H??b]gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2`R8??@??H??bmodel/Conv1D_3/conv1dhu  ?@
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2?8??@?tH?bmodel/Conv1D_3/conv1dhu ?;B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2?8??@?PH?`Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu ?;B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bmodel/Conv1D_2/Reluhu  ?B
?
hvoid fft1d_r2c_32<float, float, float2, true, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2Z8??@??H??bmodel/Conv1D_4/conv1dhu ?;B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2r8??@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu ?;B
?
dvoid fft1d_r2c_256<float, float, float2, true, false>(float2*, float const*, int3, int3, int2, int2)B??*28??@??H??bmodel/Conv1D_1/conv1dhu  B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2?8??@??H??Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu ?;B
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)9 ?* 2y8??@??H??bmodel/MaxPooling1D_2/MaxPoolhu  ?B
?
ivoid fft1d_r2c_32<float, float, float2, false, false>(float2*, float const*, int, int3, int3, int2, int2)B?"* 2Z8ߟ@??H??Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu ?;B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  2 8??@??H??b0gradient_tape/model/Conv1D_2/BiasAdd/BiasAddGradhu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2?8??@?oH?pXb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu ?;B
?
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2y8??@?kH?kXb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  H@
?
?void cudnn::cnn::im2col4d_kernel<float, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, float const*, float*)+*?2@8??@?_H?ebmodel/Conv1D_1/conv1dhu  ?B
~
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*2`8ޯ@?RH?]bmodel/Conv1D_4/conv1dhu  A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)@*?28??@??H??bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void fft1d_c2r_32<float2, float, float, false, true, false, false>(float*, float2 const*, int, int3, int3, int2, int, float, float, float*, float*)T?"* 2r8߀@??H?Abmodel/Conv1D_4/conv1dhu ?;B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?}@?(H?,bmodel/Conv1D_1/conv1dhu  ?B
?
Tvoid cask_cudnn::computeOffsetsKernel<true, false>(cask_cudnn::ComputeOffsetsParams)*?28?t@? H?+Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2?	8?s@?4H?>Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
g
volta_sgemm_32x32_sliced1x4_ntV??*?28?r@?rH?rb$gradient_tape/model/Dense_2/MatMul_1hu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?q@?qH?qb_gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInput-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
[void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, float*, float)*?2?8?q@?"H?+Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?2(8?p@?pH?pb0gradient_tape/model/Conv1D_3/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?p@?pH?pbDmodel/MaxPooling1D_2/MaxPool-0-2-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
g
volta_sgemm_32x32_sliced1x4_ntV??*?28?o@?oH?ob$gradient_tape/model/Dense_1/MatMul_1hu  HB
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?o@?oH?ob]gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?o@?oH?ob=model/Conv1D_3/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
g
volta_sgemm_32x32_sliced1x4_tnV??*?28?n@?nH?nXb"gradient_tape/model/Dense_1/MatMulhu  ?A
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?n@?nH?nb\gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGrad-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
Y
volta_sgemm_32x32_sliced1x4_nnV??*?28?m@?mH?mXbmodel/Dense_2/MatMulhu  ?A
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?l@?lH?lb^gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int),*?28?g@?gH?gbgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
Y
volta_sgemm_32x32_sliced1x4_nnV??*?28?e@?eH?eXbmodel/Dense_1/MatMulhu  ?A
?
Dcask_cudnn::computeBOffsetsKernel(cask_cudnn::ComputeBOffsetsParams)*?28?e@? H?#Xb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?c@?cH?cb^gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGrad-0-0-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *?28?`@?`H?`b/gradient_tape/model/Dense_2/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8?_@?_H?_b%gradient_tape/model/Conv1D_3/ReluGradhu  ?B
?
?void pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)9 ?*2y8?_@?_H?_bmodel/MaxPooling1D_3/MaxPoolhu??B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)3?!*?2?8?_@?_H?_b\gradient_tape/model/MaxPooling1D_2/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  2 8?Q@?QH?Qb0gradient_tape/model/Conv1D_3/BiasAdd/BiasAddGradhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?2(8?O@?OH?Obmodel/Conv1D_3/BiasAddhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?2&8?O@?OH?Ob$Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2'8?L@?LH?LXb7gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2`8?L@?LH?Lb0gradient_tape/model/Conv1D_3/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2R8?J@?JH?Jb0gradient_tape/model/Conv1D_2/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 32, 32, 7, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)<?* 2?8?I@?IH?Ib_gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInput-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  HB
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  2]8?H@?HH?Hb0gradient_tape/model/Conv1D_1/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?H@?HH?Hb%Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?2'8?F@?FH?Fb$Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?F@?FH?Fb$Adam/Adam/update_6/ResourceApplyAdamhu  ?B
^
volta_sgemm_128x32_tn9??*?28?E@?EH?EXb"gradient_tape/model/Dense_2/MatMulhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2'8?D@?DH?Dbmodel/Conv1D_3/conv1dhu  ?B
?
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*?28?C@?H?$bmodel/Conv1D_4/conv1dhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?@@?@H?@b$Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 32, 7, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)5?* 2?8?@@?@H?@b\gradient_tape/model/MaxPooling1D_3/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizerhu  HB
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?@@?@H?@b$Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?@@?@H?@b$Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 32, 32, 7, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)<?* 2?8?@@?@H?@bDmodel/MaxPooling1D_3/MaxPool-0-2-TransposeNCHWToNHWC-LayoutOptimizerhu  HB
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28??@??H??b"Adam/Adam/update/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 32, 7, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)5?* 2?8??@??H??b^gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizerhu  HB
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) x*?28??@??H??b0gradient_tape/model/Conv1D_4/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?2(8?<@?<H?<bmodel/Conv1D_3/Reluhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?;@?;H?;Xb8gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?:@?:H?:b$Adam/Adam/update_7/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?9@?9H?9b$Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?9@?9H?9Xb8gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2'8?9@?9H?9Xb8gradient_tape/model/Conv1D_3/conv1d/Conv2DBackpropFilterhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?7@?7H?7b$Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?7@?7H?7Xb7gradient_tape/model/Conv1D_2/conv1d/Conv2DBackpropInputhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?2&8?7@?7H?7bmodel/Conv1D_2/conv1dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?5@?5H?5bmodel/Conv1D_4/conv1dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?2@?2H?2Xb7gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInputhu  ?B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?0@?0H?0bAdam/addhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)#*?28?0@?0H?0b%Adam/Adam/update_11/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28?0@?0H?0bmodel/Dense_1/Reluhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)0*?28?0@?0H?0bArgMaxhu  ?B
?
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) ?*?28?0@?0H?0b/gradient_tape/model/Dense_1/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?0@?0H?0b/gradient_tape/model/Dense_1/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*?28?0@?0H?0b=model/Conv1D_4/conv1d-0-1-TransposeNCHWToNHWC-LayoutOptimizerhu  ?B
?
?void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!* 28?.@?.H?.b/gradient_tape/model/Dense_2/BiasAdd/BiasAddGradhu @?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?-@?-H?-bEqualhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?-@?-H?-bBgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*?28?,@?,H?,b]gradient_tape/model/Conv1D_4/conv1d/Conv2DBackpropInput-2-TransposeNHWCToNCHW-LayoutOptimizerhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28?,@?,H?,bmodel/Conv1D_4/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?*@?*H?*bCast_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?)@?)H?)bCast_4hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?)@?)H?)bmodel/Conv1D_1/conv1dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?)@?)H?)bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?(@?(H?(bAdam/Powhu  ?B
?
?void tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*  28?(@?(H?(b0gradient_tape/model/Conv1D_4/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?'@?'H?'bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?'@?'H?'bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?&@?&H?&bLgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanhu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int)*?28?&@?&H?&bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?%@?%H?%b/gradient_tape/model/Dense_1/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?%@?%H?%bAssignAddVariableOp_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#b%gradient_tape/model/Conv1D_4/ReluGradhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?#@?#H?#bSum_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*?28?#@?#H?#bCast_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bmodel/Conv1D_4/Reluhu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28?#@?#H?#bmodel/Dense_2/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#b
div_no_nanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*?28?#@?#H?#b`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?#@?#H?#b1sparse_categorical_crossentropy/weighted_loss/Sumhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"bAssignAddVariableOp_1hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)?!*  28?"@?"H?"b0gradient_tape/model/Conv1D_4/BiasAdd/BiasAddGradhu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?"@?"H?"bMulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"bAssignAddVariableOphu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"b3sparse_categorical_crossentropy/weighted_loss/valuehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"b?sparse_categorical_crossentropy/weighted_loss/num_elements/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?!@?!H?!bAssignAddVariableOp_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?!@?!H?!bdiv_no_nan_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?!@?!H?!bCasthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)"*?28?!@?!H?!bAdam/Cast_1hu  ?B
?
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*?28? @? H? bmodel/Dense_1/BiasAddhu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28? @? H? b
LogicalAndhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?28? @? H? b$gradient_tape/model/Dense_1/ReluGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28? @? H? bbsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*?28?@?H?Xb8gradient_tape/model/Conv1D_1/conv1d/Conv2DBackpropFilterhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?bUgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulhu  ?B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?@?H?b
Adam/Pow_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAssignAddVariableOp_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?@?H?bAdam/Adam/AssignAddVariableOphu  ?B