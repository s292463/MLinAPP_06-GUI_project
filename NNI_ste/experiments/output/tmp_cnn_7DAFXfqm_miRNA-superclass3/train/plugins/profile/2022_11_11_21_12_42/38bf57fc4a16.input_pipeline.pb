	kF��"@kF��"@!kF��"@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCkF��"@,�9$���?1���U^@A���M�?I����+@rEagerKernelExecute 0*	������f@2F
Iterator::Model9�t�yƲ?!4+��,(D@)�uŌ���?1�Y(y<�:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat*s�螥?!Ɍ667@)�?�Z�k�?1)XNa��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateE��f�R�?!z��:@)6�Ko.�?1�8��,@:Preprocessing2U
Iterator::Model::ParallelMapV2b�G,�?!?��:+@)b�G,�?1?��:+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceT^-w�?!@��e(@)T^-w�?1@��e(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�m�8)̻?!��{@��M@)�	g��ɐ?1�-��"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor������y?!�A��{@)������y?1�A��{@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�S�v;@)#��d?1uW2m`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�53.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI^d��ER@Q�n�h��:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	,�9$���?,�9$���?!,�9$���?      ��!       "	���U^@���U^@!���U^@*      ��!       2	���M�?���M�?!���M�?:	����+@����+@!����+@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q^d��ER@y�n�h��:@