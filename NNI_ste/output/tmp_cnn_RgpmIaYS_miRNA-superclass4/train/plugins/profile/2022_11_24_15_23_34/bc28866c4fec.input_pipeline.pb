	if2@if2@!if2@	�,���?�,���?!�,���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLif2@�\��m�?1R���0'@A��n���?I��f��@Y֋��hW�?rEagerKernelExecute 0*	:�O���c@2F
Iterator::Model^�Y-�Ǵ?!�=���I@)B�/h!�?1�R$���@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Hi�?!��q�;@)ׅ�O�?1	\h��7@:Preprocessing2U
Iterator::Model::ParallelMapV2�,�}�?!����'2@)�,�}�?1����'2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicezލ�A�?!M�Ѐ@)zލ�A�?1M�Ѐ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��EB[Ε?!Z6T�13+@)��~1[�?1����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���N�?!A��NrH@)Na����?1�%���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����_z?!G��L�r@)�����_z?1G��L�r@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�s���z�?!��i��.@)����ce?1�f��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�28.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�,���?IT5���A@Q�ن�`O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\��m�?�\��m�?!�\��m�?      ��!       "	R���0'@R���0'@!R���0'@*      ��!       2	��n���?��n���?!��n���?:	��f��@��f��@!��f��@B      ��!       J	֋��hW�?֋��hW�?!֋��hW�?R      ��!       Z	֋��hW�?֋��hW�?!֋��hW�?b      ��!       JGPUY�,���?b qT5���A@y�ن�`O@