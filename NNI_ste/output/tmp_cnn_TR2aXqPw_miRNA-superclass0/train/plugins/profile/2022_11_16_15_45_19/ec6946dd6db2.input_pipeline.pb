	=?Е?@=?Е?@!=?Е?@	k,f?Y@k,f?Y@!k,f?Y@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL=?Е?@???0???1?uT5@A1$'?
??I?\??@YQk?w????rEagerKernelExecute 0*	??Mb?a@2F
Iterator::Model?uʣ??!B????CG@)?^D?1u??1??q-8??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?;Nё\??!,??֛h>@)z?,C???1??d???9@:Preprocessing2U
Iterator::Model::ParallelMapV27QKs+???!????UB-@)7QKs+???1????UB-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??????!~e????@)??????1~e????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??]i???!?^rN?J@)?_w??ă?1fM}?.?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??wF[???!??BD??+@)?w??!??1????s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??q??{?!???D??@)??q??{?1???D??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:?%???!?? cuW0@)%]3?f?k?1???V?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?39.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t17.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9k,f?Y@I ?4y??L@QRo
z?B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???0??????0???!???0???      ??!       "	?uT5@?uT5@!?uT5@*      ??!       2	1$'?
??1$'?
??!1$'?
??:	?\??@?\??@!?\??@B      ??!       J	Qk?w????Qk?w????!Qk?w????R      ??!       Z	Qk?w????Qk?w????!Qk?w????b      ??!       JGPUYk,f?Y@b q ?4y??L@yRo
z?B@