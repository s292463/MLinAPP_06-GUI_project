	R??v4@R??v4@!R??v4@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'R??v4@)??q8?1s?m?B?@Iz?"n>2@r0*	.????nc@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%??1??!?x$??e>@)?1 {????1?!?@9@:Preprocessing2U
Iterator::Model::ParallelMapV2 ??????!?????2@) ??????1?????2@:Preprocessing2F
Iterator::Model4?ތ????!??MN??@@)` ??c??1?mDw?b-@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapmo?$???!G????7@)N`:?۠??1f????m,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}w+Kt??!'=?X??P@)/?$????1?o?Κ(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?h8en??!&???#@)?h8en??1&???#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??D؀?!H]=8?)@)*??D؀?1H]=8?)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?89.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?}GҒJV@Q(?mi?%@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)??q8?)??q8?!)??q8?      ??!       "	s?m?B?@s?m?B?@!s?m?B?@*      ??!       2      ??!       :	z?"n>2@z?"n>2@!z?"n>2@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?}GҒJV@y(?mi?%@