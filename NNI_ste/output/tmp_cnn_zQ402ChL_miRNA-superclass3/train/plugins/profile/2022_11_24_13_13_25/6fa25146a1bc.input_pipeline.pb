	�Yh�4s @�Yh�4s @!�Yh�4s @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�Yh�4s @�J�����?1v�և�f@A�_��D�?I�fh<@rEagerKernelExecute 0*	<�O���u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA�C��?!Y5$��Q@)���}�?1���u�CQ@:Preprocessing2F
Iterator::ModeltϺFˁ�?!,O���4@)H��0~�?19�+_**@:Preprocessing2U
Iterator::Model::ParallelMapV2@/ܹ0қ?!���!�@)@/ܹ0қ?1���!�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�_���ܓ?!݉�.m~@)�7�{�5�?1��� �@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�*Q��r�?!5�O���S@)Y�+���~?1r��#�T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��#�{?!<�l@s��?)��#�{?1<�l@s��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6����t�?!c�B��Q@)Z���аh?1��?���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorR�=�Ne?!&K(��!�?)R�=�Ne?1&K(��!�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_?!�����?)���_?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 24.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���ccR@Q���qr:@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�J�����?�J�����?!�J�����?      ��!       "	v�և�f@v�և�f@!v�և�f@*      ��!       2	�_��D�?�_��D�?!�_��D�?:	�fh<@�fh<@!�fh<@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���ccR@y���qr:@