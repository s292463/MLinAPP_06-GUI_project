	�c]�F�@�c]�F�@!�c]�F�@	������@������@!������@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�c]�F�@��]L3��?1*�Z^��@A�yUg��~?Il%t��� @Y�x\T��?rEagerKernelExecute 0*	�I+�@2U
Iterator::Model::ParallelMapV2tzލ��?!��=k�D@)tzލ��?1��=k�D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�5�Ea�?!�/?��7@)Su�l���?1�5�]�k6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ)�����?!ޣLtE�5@)J)�����?1ޣLtE�5@:Preprocessing2F
Iterator::Model)��5�?!*ss���G@)�m��ʆ�?1Dګ)<�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%̴�++�?!�-t�C:@)�?�?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!�IƩ9)9@)Z��լ3�?1Y,j����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice6\�-ˇ?!�g>S�?)6\�-ˇ?1�g>S�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�5��,�?!׌�}FJ@)�f��j+�?1��\��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�27.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9������@I���4�==@Q:�\��O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��]L3��?��]L3��?!��]L3��?      ��!       "	*�Z^��@*�Z^��@!*�Z^��@*      ��!       2	�yUg��~?�yUg��~?!�yUg��~?:	l%t��� @l%t��� @!l%t��� @B      ��!       J	�x\T��?�x\T��?!�x\T��?R      ��!       Z	�x\T��?�x\T��?!�x\T��?b      ��!       JGPUY������@b q���4�==@y:�\��O@