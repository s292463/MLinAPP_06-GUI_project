	�y ��5@�y ��5@!�y ��5@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'�y ��5@�Un2��?1��E�T@@ID�;��m2@r0*	Zd;�Osu@2U
Iterator::Model::ParallelMapV2�J %vm�?!~HDXWcH@)�J %vm�?1~HDXWcH@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���h�x�?!�.�{�L7@)m���5?�?1c��B��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���	��?!*�t��n!@)���	��?1*�t��n!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaprQ-"�ɫ?!D��d[�/@)����
�?12p�Z5c@:Preprocessing2F
Iterator::Models�ۄ{e�?!4f�*^�K@)ҋ��*��?1���6@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�r�w���?!̙zա;F@)��ԕϒ?1�� ��h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~(F�́?!_A��!B@)~(F�́?1_A��!B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�87.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��Z�2%V@Q<Q)�i�&@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Un2��?�Un2��?!�Un2��?      ��!       "	��E�T@@��E�T@@!��E�T@@*      ��!       2      ��!       :	D�;��m2@D�;��m2@!D�;��m2@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��Z�2%V@y<Q)�i�&@