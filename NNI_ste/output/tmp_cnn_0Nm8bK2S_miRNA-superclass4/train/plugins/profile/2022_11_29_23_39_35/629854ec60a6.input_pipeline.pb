	
e??k?@
e??k?@!
e??k?@	Lw\٧a@Lw\٧a@!Lw\٧a@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
e??k?@`?|x? ??1T????@AE?Ɵ?lx?I???????Y9???????rEagerKernelExecute 0*	?Zd
?@2U
Iterator::Model::ParallelMapV2??Pk?w??!?y?J@)??Pk?w??1?y?J@:Preprocessing2F
Iterator::Model????Z??!?y??LU@)n?8)?{??1zrs?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%]3?f???!??z?C @)???ܧ?1????g?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)??5??!rG?9?@)??v稓?1??????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???S????!?\?;???)???S????1?\?;???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTs??P???!?1|?!?-@)?S?K???1fb?ķ??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?;??}?!2?wd??)?;??}?12?wd??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapT?*?gz??!??)??G@)???`?Hd?1;i'????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?24.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s3.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Kw\٧a@I??)_??;@QѺ??9P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`?|x? ??`?|x? ??!`?|x? ??      ??!       "	T????@T????@!T????@*      ??!       2	E?Ɵ?lx?E?Ɵ?lx?!E?Ɵ?lx?:	??????????????!???????B      ??!       J	9???????9???????!9???????R      ??!       Z	9???????9???????!9???????b      ??!       JGPUYKw\٧a@b q??)_??;@yѺ??9P@