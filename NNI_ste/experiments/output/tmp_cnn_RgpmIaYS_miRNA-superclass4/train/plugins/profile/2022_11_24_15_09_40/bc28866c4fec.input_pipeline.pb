	���+�8@���+�8@!���+�8@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC���+�8@�"�tu�%@1N^d~=(@A�HP��?I��Q���?rEagerKernelExecute 0*	�����]a@2F
Iterator::ModelhY����?!R"�f�G@)�̰Q֧?1)Ħ��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatOWw,�I�?!�8��=@)�8~�4�?1����X�9@:Preprocessing2U
Iterator::Model::ParallelMapV2��C�r��?!�x�iK,@)��C�r��?1�x�iK,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����H�?!��-��@)�����H�?1��-��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipzn�+��?!���8�:J@)���$�?1������@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo�m��?!���t~+@)R~R�Ӂ?1�,p�Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorcb�qm�x?!\ٛEU@)cb�qm�x?1\ٛEU@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapH�Sȕz�?!8b|$H�/@)ˆ5�Eag?1�5�Lo @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 43.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP_o�ӐI@Q���,oH@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�"�tu�%@�"�tu�%@!�"�tu�%@      ��!       "	N^d~=(@N^d~=(@!N^d~=(@*      ��!       2	�HP��?�HP��?!�HP��?:	��Q���?��Q���?!��Q���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qP_o�ӐI@y���,oH@