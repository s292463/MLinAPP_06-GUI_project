	���e"rl@���e"rl@!���e"rl@	 ���$�? ���$�?! ���$�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���e"rl@ ]lZ)�?1�l��p&e@A��~��Γ?I���KUL@Y�	0,��?rEagerKernelExecute 0*	L7�A`}]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�`�HZ�?!GK6�B@)E�4~ᕤ?1�vc?�
A@:Preprocessing2F
Iterator::Model��~�?!��g��D@)��x>�?1��w>��8@:Preprocessing2U
Iterator::Model::ParallelMapV2{Nz��ړ?!śW0p0@){Nz��ړ?1śW0p0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%���?!��Fw0-@)��U��?1��H���"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipBȗP��?!@?�H�eM@)����]iy?1|w�t�	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�)��sx?!+���>@)�)��sx?1+���>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap1Xr��?!�5\�}C@)J���c?1�z*&B��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����W:_?!jyȆa��?)����W:_?1jyȆa��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice/PR`LY?!�Q�_��?)/PR`LY?1�Q�_��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�24.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���$�?I$�X÷�9@Q:�틍�R@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 ]lZ)�? ]lZ)�?! ]lZ)�?      ��!       "	�l��p&e@�l��p&e@!�l��p&e@*      ��!       2	��~��Γ?��~��Γ?!��~��Γ?:	���KUL@���KUL@!���KUL@B      ��!       J	�	0,��?�	0,��?!�	0,��?R      ��!       Z	�	0,��?�	0,��?!�	0,��?b      ��!       JGPUY���$�?b q$�X÷�9@y:�틍�R@