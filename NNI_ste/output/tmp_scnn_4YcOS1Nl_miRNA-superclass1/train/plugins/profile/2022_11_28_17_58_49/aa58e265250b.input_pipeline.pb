	??????2@??????2@!??????2@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??????2@1D??)X?@I?)X?l*0@r0*U-??!w@)      0=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?a??c??!~??O?O@)A??_???1]Ȁ??M@:Preprocessing2F
Iterator::Model?q4GV~??!?T?v2@) ??????1?GM'?A&@:Preprocessing2U
Iterator::Model::ParallelMapV2?m?8)̛?!l????V@)?m?8)̛?1l????V@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?
?rߙ?!?,??YN@)?
?rߙ?1?,??YN@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"? ?&P??!??<TbT@)*???P??1Q^?˩@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapҌE??ɨ?!?!??)*@)??3????1"??+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?X??+???!B?C?@)?X??+???1B?C?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?85.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????6oU@Q?3??I?,@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	D??)X?@D??)X?@!D??)X?@*      ??!       2      ??!       :	?)X?l*0@?)X?l*0@!?)X?l*0@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????6oU@y?3??I?,@