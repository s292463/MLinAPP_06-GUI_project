	�`��K@�`��K@!�`��K@	jY�]�F	@jY�]�F	@!jY�]�F	@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�`��K@�<,Ԛ��?1�`�;dD@I�=ϟ6z&@Y�{,}h�?r0*	Zd;�O�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-y<�?!�d�f�|U@)"�[='��?1�1�7U@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����S�?!�@��@)����*�?1� ���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�:�G�?!m�����W@)r��V�?1�A�m�@:Preprocessing2U
Iterator::Model::ParallelMapV2��ip[�?!��
I�@)��ip[�?1��
I�@:Preprocessing2F
Iterator::Model~�Ɍ���?!6i�B�@)r�t��ϑ?1~�~��?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice (��{ԏ?!~���m�?) (��{ԏ?1~���m�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ECƣ�?!� *�mH�?)��ECƣ�?1� *�mH�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�20.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9kY�]�F	@Ip����5@Q���V&�R@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�<,Ԛ��?�<,Ԛ��?!�<,Ԛ��?      ��!       "	�`�;dD@�`�;dD@!�`�;dD@*      ��!       2      ��!       :	�=ϟ6z&@�=ϟ6z&@!�=ϟ6z&@B      ��!       J	�{,}h�?�{,}h�?!�{,}h�?R      ��!       Z	�{,}h�?�{,}h�?!�{,}h�?b      ��!       JGPUYkY�]�F	@b qp����5@y���V&�R@