	l"3��-@l"3��-@!l"3��-@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:l"3��-@��!���?1]4d<J5)@I���͋��?rEagerKernelExecute 0*�rh�g@)       =2F
Iterator::Modelf���-=�?!U�R�8�G@)��6�h��?1A��tg?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�F�g�u�?!i��[�=@)o���?1�|a�Q9@:Preprocessing2U
Iterator::Model::ParallelMapV2y�t�䛝?!���W�U/@)y�t�䛝?1���W�U/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu��l�?!�{��vJ@)Z�H�s
�?1à\��#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�0�q�	�?!�__�W�@)�0�q�	�?1�__�W�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX S�?!	�I.u)@)�4f�?1.BԨM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��VBwI|?!kb����@)��VBwI|?1kb����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/o�j�?!0jr�,@)иp $h?1o��!r�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIX3�)g�.@Q����+U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��!���?��!���?!��!���?      ��!       "	]4d<J5)@]4d<J5)@!]4d<J5)@*      ��!       2      ��!       :	���͋��?���͋��?!���͋��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qX3�)g�.@y����+U@