	��R$_�2@��R$_�2@!��R$_�2@      ��!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��R$_�2@1�bԵ��@I�r0� �/@r0*	�MbXi@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7p�G�?!6.%��v>@)��%ǝҩ?1���};&9@:Preprocessing2U
Iterator::Model::ParallelMapV2�� v�С?!�%��Y1@)�� v�С?1�%��Y1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice`���Y�?!J� U�-@)`���Y�?1J� U�-@:Preprocessing2F
Iterator::Model�ǘ����?!�ݫe߽>@)�����?1�oY�L�*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap>�n�KS�?!���4�;@)�@��L�?1���)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7�֊6��?!��&�PQ@)ђ����?1'�!Nj&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor`����Ӆ?!����%B@)`����Ӆ?1����%B@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�85.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI<�ғbwU@Q ia�D,@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
      ��!             ��!       "	�bԵ��@�bԵ��@!�bԵ��@*      ��!       2      ��!       :	�r0� �/@�r0� �/@!�r0� �/@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q<�ғbwU@y ia�D,@