	�;FV4@�;FV4@!�;FV4@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�;FV4@���5�?1��ٮ�o$@I[A�K#@rEagerKernelExecute 0*	
ףp=:e@2F
Iterator::Model֩�=#�?!�{0(aI@)�.���?1���!ڷ;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��|@�3�?!gK1W�<@)À%W���?1|M�
E9@:Preprocessing2U
Iterator::Model::ParallelMapV2=�Е�?!c"?v
7@)=�Е�?1c"?v
7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev���_w�?!x�8up@)v���_w�?1x�8up@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatehB�Ēr�?!Rn���*@)Y�8��m�?1*��~@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�AB�/h�?!`���מH@)/�N[#��?1�b7�"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorv8�Jw�y?!1{���@)v8�Jw�y?11{���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�	h"lx�?!� ��q.@)i;���.h?1s�[��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�47.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�3��H@Q���eI@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���5�?���5�?!���5�?      ��!       "	��ٮ�o$@��ٮ�o$@!��ٮ�o$@*      ��!       2      ��!       :	[A�K#@[A�K#@![A�K#@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�3��H@y���eI@