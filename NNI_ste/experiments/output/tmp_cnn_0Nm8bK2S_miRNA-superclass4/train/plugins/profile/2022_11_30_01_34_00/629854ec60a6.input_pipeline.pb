	*��% �2@*��% �2@!*��% �2@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC*��% �2@��t��Y�?1��B��^0@A!���'*�?I� ��^.�?rEagerKernelExecute 0*	����xSw@2F
Iterator::Model���+�?!dY���Q@)�E�����?1�f��"P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ�f��?!2ev:E,@)�d�z�F�?1�d�%u\(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate.�5#��?!l�Q x%@)	�n��?1�V�o�@:Preprocessing2U
Iterator::Model::ParallelMapV2��j�?!R*ߜ$_@)��j�?1R*ߜ$_@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�XİÈ?!���d�	@)�XİÈ?1���d�	@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�&ݺ?!p��D�<@)�˻��?1�4���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ʦ\�}?!G�&F�?)�ʦ\�}?1G�&F�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o���?!�AOf�&@)����?g?1���RU�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�[�6_*@Q���!�U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��t��Y�?��t��Y�?!��t��Y�?      ��!       "	��B��^0@��B��^0@!��B��^0@*      ��!       2	!���'*�?!���'*�?!!���'*�?:	� ��^.�?� ��^.�?!� ��^.�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�[�6_*@y���!�U@