	qqTn��@qqTn��@!qqTn��@	&A�Mn@&A�Mn@!&A�Mn@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLqqTn��@~�$A��?1{Cr2�@A�����x?IK=By�?Y��;�Bu�?rEagerKernelExecute 0*	R���Ae@2F
Iterator::Model�U�f��?!�T 3�G@)��vLݭ?1���G&A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2v�Kp�?!}���@@)^0����?1�1[��;@:Preprocessing2U
Iterator::Model::ParallelMapV25'/2��?!�.�*@)5'/2��?1�.�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatewLݕ]0�?!��Hi��+@)�,�?2�?1LTp{��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceUl��C�?!C5!Wޑ@)Ul��C�?1C5!Wޑ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�@gҦ�?!�7���QJ@)�]�)ʥ�?1>F�ݤD@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�.R(_?!Ch�!�@)�.R(_?1Ch�!�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��U���?!��H�/@)-��;��f?1)������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�25.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&A�Mn@I�R�Eh<@Q��ڥ	�P@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~�$A��?~�$A��?!~�$A��?      ��!       "	{Cr2�@{Cr2�@!{Cr2�@*      ��!       2	�����x?�����x?!�����x?:	K=By�?K=By�?!K=By�?B      ��!       J	��;�Bu�?��;�Bu�?!��;�Bu�?R      ��!       Z	��;�Bu�?��;�Bu�?!��;�Bu�?b      ��!       JGPUY&A�Mn@b q�R�Eh<@y��ڥ	�P@