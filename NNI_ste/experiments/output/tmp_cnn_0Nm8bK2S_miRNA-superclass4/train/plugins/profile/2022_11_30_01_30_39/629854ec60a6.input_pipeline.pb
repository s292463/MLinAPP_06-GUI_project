	��Wu�2@��Wu�2@!��Wu�2@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��Wu�2@�ƃ-v{�?1*���0@A�4�;�X?I�?ޫV��?rEagerKernelExecute 0*	�ʡE��g@2F
Iterator::ModelJ}Yک��?!��#yGE@)B�f��j�?1��j�&<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatv�[����?!�[�q?@)�VC��?1�m!��;@:Preprocessing2U
Iterator::Model::ParallelMapV2�V����?!�Ҹ�`�,@)�V����?1�Ҹ�`�,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�>���?!a܆�L@)���4�?1�yvuFx$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�g�ej�?!�Q�@��@)�g�ej�?1�Q�@��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���뉮�?!�9*tk,@)��}q�J�?1�!<�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�4*p�|?!��u&�@)�4*p�|?1��u&�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapҊo(|��?!
ZC��/@)V�F�?h?1t����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI04�f�a*@Qz(sĳU@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ƃ-v{�?�ƃ-v{�?!�ƃ-v{�?      ��!       "	*���0@*���0@!*���0@*      ��!       2	�4�;�X?�4�;�X?!�4�;�X?:	�?ޫV��?�?ޫV��?!�?ޫV��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q04�f�a*@yz(sĳU@