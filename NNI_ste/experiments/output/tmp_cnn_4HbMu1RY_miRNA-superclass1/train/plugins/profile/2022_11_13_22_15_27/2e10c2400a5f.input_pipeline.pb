	��$��P@��$��P@!��$��P@	zf�#
@zf�#
@!zf�#
@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��$��P@Ot]��y�?1��%��?A��A�F�?Iu��.�@Yr��[v�?rEagerKernelExecute 0*	p����x@2U
Iterator::Model::ParallelMapV2vR_�vj�?!���Ea�K@)vR_�vj�?1���Ea�K@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatޒ��ɳ?!?�"JZu3@)�n�UfJ�?1�z�}�*@:Preprocessing2F
Iterator::ModelH�]��-�?! ����P@)i o�ŧ?1��؍�_'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�m3⑘?!ջ-0)@)�m3⑘?1ջ-0)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicek���T�?!���
��@)k���T�?1���
��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�c���Ȥ?!�%Wnp$@)�'֩�=�?1"[����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Z&��|�?!�Ӯ֯6@@)q�GR�À?1��ab| @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�R�{/�?!�6.��%@)����gf?1���m=�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�57.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9zf�#
@Ibւ���P@Q��Qg)>@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ot]��y�?Ot]��y�?!Ot]��y�?      ��!       "	��%��?��%��?!��%��?*      ��!       2	��A�F�?��A�F�?!��A�F�?:	u��.�@u��.�@!u��.�@B      ��!       J	r��[v�?r��[v�?!r��[v�?R      ��!       Z	r��[v�?r��[v�?!r��[v�?b      ��!       JGPUYzf�#
@b qbւ���P@y��Qg)>@