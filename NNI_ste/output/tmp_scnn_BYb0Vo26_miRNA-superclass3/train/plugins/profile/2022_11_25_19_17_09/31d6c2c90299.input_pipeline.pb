	%"���Q-@%"���Q-@!%"���Q-@      ��!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails%"���Q-@1���`C@I��G���(@r0*	V-�mw@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatm������?!�U�{ԔP@)-��m�?1r�ꫭ�O@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!-NK�1@)9�Z��v�?1���mh'@:Preprocessing2F
Iterator::Model��C5%Y�?!�xI��T(@)>w��׹�?1�����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice������?!=#�޲@)������?1=#�޲@:Preprocessing2U
Iterator::Model::ParallelMapV2�mr��?!����G�@)�mr��?1����G�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���^�?!���Hm�U@)M
J�ʍ?1��K�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��1>�^�?!5u[��O@)��1>�^�?15u[��O@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�84.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI$�
�UU@Q�v�S%/@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
      ��!             ��!       "	���`C@���`C@!���`C@*      ��!       2      ��!       :	��G���(@��G���(@!��G���(@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q$�
�UU@y�v�S%/@