??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
~
Conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv1D_1/kernel
w
#Conv1D_1/kernel/Read/ReadVariableOpReadVariableOpConv1D_1/kernel*"
_output_shapes
:*
dtype0
r
Conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv1D_1/bias
k
!Conv1D_1/bias/Read/ReadVariableOpReadVariableOpConv1D_1/bias*
_output_shapes
:*
dtype0
~
Conv1D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:V* 
shared_nameConv1D_2/kernel
w
#Conv1D_2/kernel/Read/ReadVariableOpReadVariableOpConv1D_2/kernel*"
_output_shapes
:V*
dtype0
r
Conv1D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*
shared_nameConv1D_2/bias
k
!Conv1D_2/bias/Read/ReadVariableOpReadVariableOpConv1D_2/bias*
_output_shapes
:V*
dtype0
~
Conv1D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:V/* 
shared_nameConv1D_3/kernel
w
#Conv1D_3/kernel/Read/ReadVariableOpReadVariableOpConv1D_3/kernel*"
_output_shapes
:V/*
dtype0
r
Conv1D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*
shared_nameConv1D_3/bias
k
!Conv1D_3/bias/Read/ReadVariableOpReadVariableOpConv1D_3/bias*
_output_shapes
:/*
dtype0
~
Conv1D_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:/#* 
shared_nameConv1D_4/kernel
w
#Conv1D_4/kernel/Read/ReadVariableOpReadVariableOpConv1D_4/kernel*"
_output_shapes
:/#*
dtype0
r
Conv1D_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*
shared_nameConv1D_4/bias
k
!Conv1D_4/bias/Read/ReadVariableOpReadVariableOpConv1D_4/bias*
_output_shapes
:#*
dtype0
y
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?y*
shared_nameDense_1/kernel
r
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes
:	?y*
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:y*
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
:y*
dtype0
x
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:y!*
shared_nameDense_2/kernel
q
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes

:y!*
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
:!*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv1D_1/kernel/m
?
*Adam/Conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/m*"
_output_shapes
:*
dtype0
?
Adam/Conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_1/bias/m
y
(Adam/Conv1D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv1D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*'
shared_nameAdam/Conv1D_2/kernel/m
?
*Adam/Conv1D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/m*"
_output_shapes
:V*
dtype0
?
Adam/Conv1D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*%
shared_nameAdam/Conv1D_2/bias/m
y
(Adam/Conv1D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/m*
_output_shapes
:V*
dtype0
?
Adam/Conv1D_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:V/*'
shared_nameAdam/Conv1D_3/kernel/m
?
*Adam/Conv1D_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/m*"
_output_shapes
:V/*
dtype0
?
Adam/Conv1D_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/Conv1D_3/bias/m
y
(Adam/Conv1D_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/m*
_output_shapes
:/*
dtype0
?
Adam/Conv1D_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/#*'
shared_nameAdam/Conv1D_4/kernel/m
?
*Adam/Conv1D_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/m*"
_output_shapes
:/#*
dtype0
?
Adam/Conv1D_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*%
shared_nameAdam/Conv1D_4/bias/m
y
(Adam/Conv1D_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/m*
_output_shapes
:#*
dtype0
?
Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?y*&
shared_nameAdam/Dense_1/kernel/m
?
)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m*
_output_shapes
:	?y*
dtype0
~
Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:y*$
shared_nameAdam/Dense_1/bias/m
w
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes
:y*
dtype0
?
Adam/Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:y!*&
shared_nameAdam/Dense_2/kernel/m

)Adam/Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/m*
_output_shapes

:y!*
dtype0
~
Adam/Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/m
w
'Adam/Dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/m*
_output_shapes
:!*
dtype0
?
Adam/Conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv1D_1/kernel/v
?
*Adam/Conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/v*"
_output_shapes
:*
dtype0
?
Adam/Conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_1/bias/v
y
(Adam/Conv1D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/Conv1D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*'
shared_nameAdam/Conv1D_2/kernel/v
?
*Adam/Conv1D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/v*"
_output_shapes
:V*
dtype0
?
Adam/Conv1D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V*%
shared_nameAdam/Conv1D_2/bias/v
y
(Adam/Conv1D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/v*
_output_shapes
:V*
dtype0
?
Adam/Conv1D_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:V/*'
shared_nameAdam/Conv1D_3/kernel/v
?
*Adam/Conv1D_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/v*"
_output_shapes
:V/*
dtype0
?
Adam/Conv1D_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*%
shared_nameAdam/Conv1D_3/bias/v
y
(Adam/Conv1D_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/v*
_output_shapes
:/*
dtype0
?
Adam/Conv1D_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/#*'
shared_nameAdam/Conv1D_4/kernel/v
?
*Adam/Conv1D_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/v*"
_output_shapes
:/#*
dtype0
?
Adam/Conv1D_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:#*%
shared_nameAdam/Conv1D_4/bias/v
y
(Adam/Conv1D_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/v*
_output_shapes
:#*
dtype0
?
Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?y*&
shared_nameAdam/Dense_1/kernel/v
?
)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v*
_output_shapes
:	?y*
dtype0
~
Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:y*$
shared_nameAdam/Dense_1/bias/v
w
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes
:y*
dtype0
?
Adam/Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:y!*&
shared_nameAdam/Dense_2/kernel/v

)Adam/Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/v*
_output_shapes

:y!*
dtype0
~
Adam/Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/v
w
'Adam/Dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/v*
_output_shapes
:!*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
R
;trainable_variables
<	variables
=regularization_losses
>	keras_api
h

?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
R
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?1m?2m??m?@m?Im?Jm?Om?Pm?v?v?#v?$v?1v?2v??v?@v?Iv?Jv?Ov?Pv?
V
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11
 
V
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11
?
trainable_variables
Zmetrics
regularization_losses
[non_trainable_variables
	variables

\layers
]layer_metrics
^layer_regularization_losses
 
[Y
VARIABLE_VALUEConv1D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
_metrics
	variables
`non_trainable_variables
regularization_losses

alayers
blayer_metrics
clayer_regularization_losses
 
 
 
?
trainable_variables
dmetrics
	variables
enon_trainable_variables
regularization_losses

flayers
glayer_metrics
hlayer_regularization_losses
 
 
 
?
trainable_variables
imetrics
 	variables
jnon_trainable_variables
!regularization_losses

klayers
llayer_metrics
mlayer_regularization_losses
[Y
VARIABLE_VALUEConv1D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
%trainable_variables
nmetrics
&	variables
onon_trainable_variables
'regularization_losses

players
qlayer_metrics
rlayer_regularization_losses
 
 
 
?
)trainable_variables
smetrics
*	variables
tnon_trainable_variables
+regularization_losses

ulayers
vlayer_metrics
wlayer_regularization_losses
 
 
 
?
-trainable_variables
xmetrics
.	variables
ynon_trainable_variables
/regularization_losses

zlayers
{layer_metrics
|layer_regularization_losses
[Y
VARIABLE_VALUEConv1D_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
3trainable_variables
}metrics
4	variables
~non_trainable_variables
5regularization_losses

layers
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
7trainable_variables
?metrics
8	variables
?non_trainable_variables
9regularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
;trainable_variables
?metrics
<	variables
?non_trainable_variables
=regularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
[Y
VARIABLE_VALUEConv1D_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
?
Atrainable_variables
?metrics
B	variables
?non_trainable_variables
Cregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
 
 
 
?
Etrainable_variables
?metrics
F	variables
?non_trainable_variables
Gregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
Ktrainable_variables
?metrics
L	variables
?non_trainable_variables
Mregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
?
Qtrainable_variables
?metrics
R	variables
?non_trainable_variables
Sregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_13247
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv1D_1/kernel/Read/ReadVariableOp!Conv1D_1/bias/Read/ReadVariableOp#Conv1D_2/kernel/Read/ReadVariableOp!Conv1D_2/bias/Read/ReadVariableOp#Conv1D_3/kernel/Read/ReadVariableOp!Conv1D_3/bias/Read/ReadVariableOp#Conv1D_4/kernel/Read/ReadVariableOp!Conv1D_4/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp"Dense_2/kernel/Read/ReadVariableOp Dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv1D_1/kernel/m/Read/ReadVariableOp(Adam/Conv1D_1/bias/m/Read/ReadVariableOp*Adam/Conv1D_2/kernel/m/Read/ReadVariableOp(Adam/Conv1D_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_3/kernel/m/Read/ReadVariableOp(Adam/Conv1D_3/bias/m/Read/ReadVariableOp*Adam/Conv1D_4/kernel/m/Read/ReadVariableOp(Adam/Conv1D_4/bias/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp)Adam/Dense_2/kernel/m/Read/ReadVariableOp'Adam/Dense_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_1/kernel/v/Read/ReadVariableOp(Adam/Conv1D_1/bias/v/Read/ReadVariableOp*Adam/Conv1D_2/kernel/v/Read/ReadVariableOp(Adam/Conv1D_2/bias/v/Read/ReadVariableOp*Adam/Conv1D_3/kernel/v/Read/ReadVariableOp(Adam/Conv1D_3/bias/v/Read/ReadVariableOp*Adam/Conv1D_4/kernel/v/Read/ReadVariableOp(Adam/Conv1D_4/bias/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp)Adam/Dense_2/kernel/v/Read/ReadVariableOp'Adam/Dense_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_13882
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv1D_1/kernel/mAdam/Conv1D_1/bias/mAdam/Conv1D_2/kernel/mAdam/Conv1D_2/bias/mAdam/Conv1D_3/kernel/mAdam/Conv1D_3/bias/mAdam/Conv1D_4/kernel/mAdam/Conv1D_4/bias/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_2/kernel/mAdam/Dense_2/bias/mAdam/Conv1D_1/kernel/vAdam/Conv1D_1/bias/vAdam/Conv1D_2/kernel/vAdam/Conv1D_2/bias/vAdam/Conv1D_3/kernel/vAdam/Conv1D_3/bias/vAdam/Conv1D_4/kernel/vAdam/Conv1D_4/bias/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_2/kernel/vAdam/Dense_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_14027??

?;
?
@__inference_model_layer_call_and_return_conditional_losses_13210
input_1$
conv1d_1_13172:
conv1d_1_13174:$
conv1d_2_13179:V
conv1d_2_13181:V$
conv1d_3_13186:V/
conv1d_3_13188:/$
conv1d_4_13193:/#
conv1d_4_13195:# 
dense_1_13199:	?y
dense_1_13201:y
dense_2_13204:y!
dense_2_13206:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_13172conv1d_1_13174*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_126992"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_126402 
MaxPooling1D_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_129872!
dropout/StatefulPartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_13179conv1d_2_13181*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_127292"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_126552 
MaxPooling1D_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_129542#
!dropout_1/StatefulPartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_13186conv1d_3_13188*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????"/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_127592"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_126702 
MaxPooling1D_3/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_129212#
!dropout_2/StatefulPartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_13193conv1d_4_13195*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_127892"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_128012
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_13199dense_1_13201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_128142!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13204dense_2_13206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_128312!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?
!__inference__traced_restore_14027
file_prefix6
 assignvariableop_conv1d_1_kernel:.
 assignvariableop_1_conv1d_1_bias:8
"assignvariableop_2_conv1d_2_kernel:V.
 assignvariableop_3_conv1d_2_bias:V8
"assignvariableop_4_conv1d_3_kernel:V/.
 assignvariableop_5_conv1d_3_bias:/8
"assignvariableop_6_conv1d_4_kernel:/#.
 assignvariableop_7_conv1d_4_bias:#4
!assignvariableop_8_dense_1_kernel:	?y-
assignvariableop_9_dense_1_bias:y4
"assignvariableop_10_dense_2_kernel:y!.
 assignvariableop_11_dense_2_bias:!'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: @
*assignvariableop_21_adam_conv1d_1_kernel_m:6
(assignvariableop_22_adam_conv1d_1_bias_m:@
*assignvariableop_23_adam_conv1d_2_kernel_m:V6
(assignvariableop_24_adam_conv1d_2_bias_m:V@
*assignvariableop_25_adam_conv1d_3_kernel_m:V/6
(assignvariableop_26_adam_conv1d_3_bias_m:/@
*assignvariableop_27_adam_conv1d_4_kernel_m:/#6
(assignvariableop_28_adam_conv1d_4_bias_m:#<
)assignvariableop_29_adam_dense_1_kernel_m:	?y5
'assignvariableop_30_adam_dense_1_bias_m:y;
)assignvariableop_31_adam_dense_2_kernel_m:y!5
'assignvariableop_32_adam_dense_2_bias_m:!@
*assignvariableop_33_adam_conv1d_1_kernel_v:6
(assignvariableop_34_adam_conv1d_1_bias_v:@
*assignvariableop_35_adam_conv1d_2_kernel_v:V6
(assignvariableop_36_adam_conv1d_2_bias_v:V@
*assignvariableop_37_adam_conv1d_3_kernel_v:V/6
(assignvariableop_38_adam_conv1d_3_bias_v:/@
*assignvariableop_39_adam_conv1d_4_kernel_v:/#6
(assignvariableop_40_adam_conv1d_4_bias_v:#<
)assignvariableop_41_adam_dense_1_kernel_v:	?y5
'assignvariableop_42_adam_dense_1_bias_v:y;
)assignvariableop_43_adam_dense_2_kernel_v:y!5
'assignvariableop_44_adam_dense_2_bias_v:!
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ۍ
?

@__inference_model_layer_call_and_return_conditional_losses_13492

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:V6
(conv1d_2_biasadd_readvariableop_resource:VJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:V/6
(conv1d_3_biasadd_readvariableop_resource:/J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:/#6
(conv1d_4_biasadd_readvariableop_resource:#9
&dense_1_matmul_readvariableop_resource:	?y5
'dense_1_biasadd_readvariableop_resource:y8
&dense_2_matmul_readvariableop_resource:y!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
MaxPooling1D_1/Squeezes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *a͜?2
dropout/dropout/Const?
dropout/dropout/MulMulMaxPooling1D_1/Squeeze:output:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/Mul}
dropout/dropout/ShapeShapeMaxPooling1D_1/Squeeze:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0*
seed2????2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?<>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/dropout/Mul_1?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????V*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????V*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????V2
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????V2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????V2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????$V*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????$V*
squeeze_dims
2
MaxPooling1D_2/Squeezew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *@???2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulMaxPooling1D_2/Squeeze:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????$V2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapeMaxPooling1D_2/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????$V*
dtype0*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????$V2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????$V2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????$V2
dropout_1/dropout/Mul_1?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????$V2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V/*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V/2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????"/*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????"/*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????"/2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????"/2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????"/2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2
MaxPooling1D_3/Squeezew
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?`??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulMaxPooling1D_3/Squeeze:output:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????/2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShapeMaxPooling1D_3/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????/*
dtype0*
seed220
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *L?9>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????/2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????/2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????/2
dropout_2/dropout/Mul_1?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsdropout_2/dropout/Mul_1:z:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????/2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:/#*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:/#2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?y*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:y!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_2_layer_call_fn_12661

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_126552
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_13631

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_129212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_13522

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127112
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_13128
input_1
unknown:
	unknown_0:
	unknown_1:V
	unknown_2:V
	unknown_3:V/
	unknown_4:/
	unknown_5:/#
	unknown_6:#
	unknown_7:	?y
	unknown_8:y
	unknown_9:y!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_130722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?\
?
__inference__traced_save_13882
file_prefix.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::V:V:V/:/:/#:#:	?y:y:y!:!: : : : : : : : : :::V:V:V/:/:/#:#:	?y:y:y!:!:::V:V:V/:/:/#:#:	?y:y:y!:!: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:V: 

_output_shapes
:V:($
"
_output_shapes
:V/: 

_output_shapes
:/:($
"
_output_shapes
:/#: 

_output_shapes
:#:%	!

_output_shapes
:	?y: 


_output_shapes
:y:$ 

_output_shapes

:y!: 

_output_shapes
:!:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:V: 

_output_shapes
:V:($
"
_output_shapes
:V/: 

_output_shapes
:/:($
"
_output_shapes
:/#: 

_output_shapes
:#:%!

_output_shapes
:	?y: 

_output_shapes
:y:$  

_output_shapes

:y!: !

_output_shapes
:!:("$
"
_output_shapes
:: #

_output_shapes
::($$
"
_output_shapes
:V: %

_output_shapes
:V:(&$
"
_output_shapes
:V/: '

_output_shapes
:/:(($
"
_output_shapes
:/#: )

_output_shapes
:#:%*!

_output_shapes
:	?y: +

_output_shapes
:y:$, 

_output_shapes

:y!: -

_output_shapes
:!:.

_output_shapes
: 
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13704

inputs1
matmul_readvariableop_resource:	?y-
biasadd_readvariableop_resource:y
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?y*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:y*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????y2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????y2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?q
?

@__inference_model_layer_call_and_return_conditional_losses_13388

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:6
(conv1d_1_biasadd_readvariableop_resource:J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:V6
(conv1d_2_biasadd_readvariableop_resource:VJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:V/6
(conv1d_3_biasadd_readvariableop_resource:/J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:/#6
(conv1d_4_biasadd_readvariableop_resource:#9
&dense_1_matmul_readvariableop_resource:	?y5
'dense_1_biasadd_readvariableop_resource:y8
&dense_2_matmul_readvariableop_resource:y!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
dropout/IdentityIdentityMaxPooling1D_1/Squeeze:output:0*
T0*,
_output_shapes
:??????????2
dropout/Identity?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsdropout/Identity:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????V*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????V*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????V2
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????V2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????V2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????$V*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????$V*
squeeze_dims
2
MaxPooling1D_2/Squeeze?
dropout_1/IdentityIdentityMaxPooling1D_2/Squeeze:output:0*
T0*+
_output_shapes
:?????????$V2
dropout_1/Identity?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsdropout_1/Identity:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????$V2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V/*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V/2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????"/*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????"/*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????"/2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????"/2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????"/2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2
MaxPooling1D_3/Squeeze?
dropout_2/IdentityIdentityMaxPooling1D_3/Squeeze:output:0*
T0*+
_output_shapes
:?????????/2
dropout_2/Identity?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsdropout_2/Identity:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????/2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:/#*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:/#2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?y*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:y!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_12699

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_12987

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *a͜?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?<>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_4_layer_call_fn_13657

inputs
unknown:/#
	unknown_0:#
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_127892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????/: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_12729

inputsA
+conv1d_expanddims_1_readvariableop_resource:V-
biasadd_readvariableop_resource:V
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????V*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????V*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????V2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????V2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_1_layer_call_fn_13501

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_126992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_13596

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *@???2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????$V2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????$V*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????$V2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????$V2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????$V2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????$V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_12801

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12640

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_12954

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *@???2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????$V2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????$V*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????$V2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????$V2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????$V2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????$V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_1_layer_call_fn_12646

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_126402
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_13527

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_129872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_13579

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_129542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????$V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_13684

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_13626

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_127712
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12655

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_12711

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_13621

inputsA
+conv1d_expanddims_1_readvariableop_resource:V/-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????$V2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V/*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V/2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????"/*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????"/*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????"/2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????"/2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????"/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$V: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_13673

inputsA
+conv1d_expanddims_1_readvariableop_resource:/#-
biasadd_readvariableop_resource:#
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????/2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:/#*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:/#2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_13517

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_12759

inputsA
+conv1d_expanddims_1_readvariableop_resource:V/-
biasadd_readvariableop_resource:/
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????$V2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V/*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V/2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????"/*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????"/*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????"/2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????"/2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????"/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$V: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_12814

inputs1
matmul_readvariableop_resource:	?y-
biasadd_readvariableop_resource:y
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?y*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:y*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????y2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????y2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_13569

inputsA
+conv1d_expanddims_1_readvariableop_resource:V-
biasadd_readvariableop_resource:V
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????V*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????V*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:V*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????V2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????V2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_3_layer_call_fn_12676

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_126702
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_13636

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????/2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????/2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_2_layer_call_fn_13553

inputs
unknown:V
	unknown_0:V
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_127292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12670

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_13247
input_1
unknown:
	unknown_0:
	unknown_1:V
	unknown_2:V
	unknown_3:V/
	unknown_4:/
	unknown_5:/#
	unknown_6:#
	unknown_7:	?y
	unknown_8:y
	unknown_9:y!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_126312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_13724

inputs0
matmul_readvariableop_resource:y!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:y!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????y: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?6
?
@__inference_model_layer_call_and_return_conditional_losses_12838

inputs$
conv1d_1_12700:
conv1d_1_12702:$
conv1d_2_12730:V
conv1d_2_12732:V$
conv1d_3_12760:V/
conv1d_3_12762:/$
conv1d_4_12790:/#
conv1d_4_12792:# 
dense_1_12815:	?y
dense_1_12817:y
dense_2_12832:y!
dense_2_12834:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_12700conv1d_1_12702*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_126992"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_126402 
MaxPooling1D_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127112
dropout/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_12730conv1d_2_12732*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_127292"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_126552 
MaxPooling1D_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_127412
dropout_1/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_12760conv1d_3_12762*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????"/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_127592"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_126702 
MaxPooling1D_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_127712
dropout_2/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_12790conv1d_4_12792*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_127892"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_128012
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_12815dense_1_12817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_128142!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_12832dense_2_12834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_128312!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?}
?
 __inference__wrapped_model_12631
input_1P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:<
.model_conv1d_1_biasadd_readvariableop_resource:P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:V<
.model_conv1d_2_biasadd_readvariableop_resource:VP
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:V/<
.model_conv1d_3_biasadd_readvariableop_resource:/P
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:/#<
.model_conv1d_4_biasadd_readvariableop_resource:#?
,model_dense_1_matmul_readvariableop_resource:	?y;
-model_dense_1_biasadd_readvariableop_resource:y>
,model_dense_2_matmul_readvariableop_resource:y!;
-model_dense_2_biasadd_readvariableop_resource:!
identity??%model/Conv1D_1/BiasAdd/ReadVariableOp?1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_2/BiasAdd/ReadVariableOp?1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_3/BiasAdd/ReadVariableOp?1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_4/BiasAdd/ReadVariableOp?1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?$model/Dense_1/BiasAdd/ReadVariableOp?#model/Dense_1/MatMul/ReadVariableOp?$model/Dense_2/BiasAdd/ReadVariableOp?#model/Dense_2/MatMul/ReadVariableOp?
$model/Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_1/conv1d/ExpandDims/dim?
 model/Conv1D_1/conv1d/ExpandDims
ExpandDimsinput_1-model/Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2"
 model/Conv1D_1/conv1d/ExpandDims?
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_1/conv1d/ExpandDims_1/dim?
"model/Conv1D_1/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"model/Conv1D_1/conv1d/ExpandDims_1?
model/Conv1D_1/conv1dConv2D)model/Conv1D_1/conv1d/ExpandDims:output:0+model/Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/Conv1D_1/conv1d?
model/Conv1D_1/conv1d/SqueezeSqueezemodel/Conv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
model/Conv1D_1/conv1d/Squeeze?
%model/Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/Conv1D_1/BiasAdd/ReadVariableOp?
model/Conv1D_1/BiasAddBiasAdd&model/Conv1D_1/conv1d/Squeeze:output:0-model/Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model/Conv1D_1/BiasAdd?
model/Conv1D_1/ReluRelumodel/Conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model/Conv1D_1/Relu?
#model/MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_1/ExpandDims/dim?
model/MaxPooling1D_1/ExpandDims
ExpandDims!model/Conv1D_1/Relu:activations:0,model/MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2!
model/MaxPooling1D_1/ExpandDims?
model/MaxPooling1D_1/MaxPoolMaxPool(model/MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_1/MaxPool?
model/MaxPooling1D_1/SqueezeSqueeze%model/MaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims
2
model/MaxPooling1D_1/Squeeze?
model/dropout/IdentityIdentity%model/MaxPooling1D_1/Squeeze:output:0*
T0*,
_output_shapes
:??????????2
model/dropout/Identity?
$model/Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_2/conv1d/ExpandDims/dim?
 model/Conv1D_2/conv1d/ExpandDims
ExpandDimsmodel/dropout/Identity:output:0-model/Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2"
 model/Conv1D_2/conv1d/ExpandDims?
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V*
dtype023
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_2/conv1d/ExpandDims_1/dim?
"model/Conv1D_2/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V2$
"model/Conv1D_2/conv1d/ExpandDims_1?
model/Conv1D_2/conv1dConv2D)model/Conv1D_2/conv1d/ExpandDims:output:0+model/Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????V*
paddingVALID*
strides
2
model/Conv1D_2/conv1d?
model/Conv1D_2/conv1d/SqueezeSqueezemodel/Conv1D_2/conv1d:output:0*
T0*,
_output_shapes
:??????????V*
squeeze_dims

?????????2
model/Conv1D_2/conv1d/Squeeze?
%model/Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:V*
dtype02'
%model/Conv1D_2/BiasAdd/ReadVariableOp?
model/Conv1D_2/BiasAddBiasAdd&model/Conv1D_2/conv1d/Squeeze:output:0-model/Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????V2
model/Conv1D_2/BiasAdd?
model/Conv1D_2/ReluRelumodel/Conv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????V2
model/Conv1D_2/Relu?
#model/MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_2/ExpandDims/dim?
model/MaxPooling1D_2/ExpandDims
ExpandDims!model/Conv1D_2/Relu:activations:0,model/MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????V2!
model/MaxPooling1D_2/ExpandDims?
model/MaxPooling1D_2/MaxPoolMaxPool(model/MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????$V*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_2/MaxPool?
model/MaxPooling1D_2/SqueezeSqueeze%model/MaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????$V*
squeeze_dims
2
model/MaxPooling1D_2/Squeeze?
model/dropout_1/IdentityIdentity%model/MaxPooling1D_2/Squeeze:output:0*
T0*+
_output_shapes
:?????????$V2
model/dropout_1/Identity?
$model/Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_3/conv1d/ExpandDims/dim?
 model/Conv1D_3/conv1d/ExpandDims
ExpandDims!model/dropout_1/Identity:output:0-model/Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????$V2"
 model/Conv1D_3/conv1d/ExpandDims?
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:V/*
dtype023
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_3/conv1d/ExpandDims_1/dim?
"model/Conv1D_3/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:V/2$
"model/Conv1D_3/conv1d/ExpandDims_1?
model/Conv1D_3/conv1dConv2D)model/Conv1D_3/conv1d/ExpandDims:output:0+model/Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????"/*
paddingVALID*
strides
2
model/Conv1D_3/conv1d?
model/Conv1D_3/conv1d/SqueezeSqueezemodel/Conv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????"/*
squeeze_dims

?????????2
model/Conv1D_3/conv1d/Squeeze?
%model/Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype02'
%model/Conv1D_3/BiasAdd/ReadVariableOp?
model/Conv1D_3/BiasAddBiasAdd&model/Conv1D_3/conv1d/Squeeze:output:0-model/Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????"/2
model/Conv1D_3/BiasAdd?
model/Conv1D_3/ReluRelumodel/Conv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????"/2
model/Conv1D_3/Relu?
#model/MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_3/ExpandDims/dim?
model/MaxPooling1D_3/ExpandDims
ExpandDims!model/Conv1D_3/Relu:activations:0,model/MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????"/2!
model/MaxPooling1D_3/ExpandDims?
model/MaxPooling1D_3/MaxPoolMaxPool(model/MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????/*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_3/MaxPool?
model/MaxPooling1D_3/SqueezeSqueeze%model/MaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????/*
squeeze_dims
2
model/MaxPooling1D_3/Squeeze?
model/dropout_2/IdentityIdentity%model/MaxPooling1D_3/Squeeze:output:0*
T0*+
_output_shapes
:?????????/2
model/dropout_2/Identity?
$model/Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_4/conv1d/ExpandDims/dim?
 model/Conv1D_4/conv1d/ExpandDims
ExpandDims!model/dropout_2/Identity:output:0-model/Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????/2"
 model/Conv1D_4/conv1d/ExpandDims?
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:/#*
dtype023
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_4/conv1d/ExpandDims_1/dim?
"model/Conv1D_4/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:/#2$
"model/Conv1D_4/conv1d/ExpandDims_1?
model/Conv1D_4/conv1dConv2D)model/Conv1D_4/conv1d/ExpandDims:output:0+model/Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
2
model/Conv1D_4/conv1d?
model/Conv1D_4/conv1d/SqueezeSqueezemodel/Conv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
model/Conv1D_4/conv1d/Squeeze?
%model/Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:#*
dtype02'
%model/Conv1D_4/BiasAdd/ReadVariableOp?
model/Conv1D_4/BiasAddBiasAdd&model/Conv1D_4/conv1d/Squeeze:output:0-model/Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2
model/Conv1D_4/BiasAdd?
model/Conv1D_4/ReluRelumodel/Conv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????#2
model/Conv1D_4/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
model/flatten/Const?
model/flatten/ReshapeReshape!model/Conv1D_4/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshape?
#model/Dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?y*
dtype02%
#model/Dense_1/MatMul/ReadVariableOp?
model/Dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
model/Dense_1/MatMul?
$model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype02&
$model/Dense_1/BiasAdd/ReadVariableOp?
model/Dense_1/BiasAddBiasAddmodel/Dense_1/MatMul:product:0,model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y2
model/Dense_1/BiasAdd?
model/Dense_1/ReluRelumodel/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y2
model/Dense_1/Relu?
#model/Dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:y!*
dtype02%
#model/Dense_2/MatMul/ReadVariableOp?
model/Dense_2/MatMulMatMul model/Dense_1/Relu:activations:0+model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/MatMul?
$model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02&
$model/Dense_2/BiasAdd/ReadVariableOp?
model/Dense_2/BiasAddBiasAddmodel/Dense_2/MatMul:product:0,model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/BiasAdd?
model/Dense_2/SoftmaxSoftmaxmodel/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/Softmax?
IdentityIdentitymodel/Dense_2/Softmax:softmax:0&^model/Conv1D_1/BiasAdd/ReadVariableOp2^model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_2/BiasAdd/ReadVariableOp2^model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_3/BiasAdd/ReadVariableOp2^model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_4/BiasAdd/ReadVariableOp2^model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp%^model/Dense_1/BiasAdd/ReadVariableOp$^model/Dense_1/MatMul/ReadVariableOp%^model/Dense_2/BiasAdd/ReadVariableOp$^model/Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2N
%model/Conv1D_1/BiasAdd/ReadVariableOp%model/Conv1D_1/BiasAdd/ReadVariableOp2f
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_2/BiasAdd/ReadVariableOp%model/Conv1D_2/BiasAdd/ReadVariableOp2f
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_3/BiasAdd/ReadVariableOp%model/Conv1D_3/BiasAdd/ReadVariableOp2f
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_4/BiasAdd/ReadVariableOp%model/Conv1D_4/BiasAdd/ReadVariableOp2f
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2L
$model/Dense_1/BiasAdd/ReadVariableOp$model/Dense_1/BiasAdd/ReadVariableOp2J
#model/Dense_1/MatMul/ReadVariableOp#model/Dense_1/MatMul/ReadVariableOp2L
$model/Dense_2/BiasAdd/ReadVariableOp$model/Dense_2/BiasAdd/ReadVariableOp2J
#model/Dense_2/MatMul/ReadVariableOp#model/Dense_2/MatMul/ReadVariableOp:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_12921

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?`??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????/2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????/*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *L?9>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????/2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????/2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????/2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_3_layer_call_fn_13605

inputs
unknown:V/
	unknown_0:/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????"/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_127592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????"/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????$V: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_13584

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????$V2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????$V2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_13544

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *a͜?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?<>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_2_layer_call_fn_13713

inputs
unknown:y!
	unknown_0:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_128312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????y: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?6
?
@__inference_model_layer_call_and_return_conditional_losses_13169
input_1$
conv1d_1_13131:
conv1d_1_13133:$
conv1d_2_13138:V
conv1d_2_13140:V$
conv1d_3_13145:V/
conv1d_3_13147:/$
conv1d_4_13152:/#
conv1d_4_13154:# 
dense_1_13158:	?y
dense_1_13160:y
dense_2_13163:y!
dense_2_13165:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_13131conv1d_1_13133*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_126992"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_126402 
MaxPooling1D_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127112
dropout/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_13138conv1d_2_13140*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_127292"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_126552 
MaxPooling1D_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_127412
dropout_1/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_13145conv1d_3_13147*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????"/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_127592"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_126702 
MaxPooling1D_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_127712
dropout_2/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_13152conv1d_4_13154*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_127892"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_128012
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_13158dense_1_13160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_128142!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13163dense_2_13165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_128312!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_13648

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?`??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????/2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????/*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *L?9>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????/2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????/2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????/2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????/2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_13678

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_128012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_12741

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????$V2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????$V2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?;
?
@__inference_model_layer_call_and_return_conditional_losses_13072

inputs$
conv1d_1_13034:
conv1d_1_13036:$
conv1d_2_13041:V
conv1d_2_13043:V$
conv1d_3_13048:V/
conv1d_3_13050:/$
conv1d_4_13055:/#
conv1d_4_13057:# 
dense_1_13061:	?y
dense_1_13063:y
dense_2_13066:y!
dense_2_13068:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_13034conv1d_1_13036*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_126992"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_126402 
MaxPooling1D_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_129872!
dropout/StatefulPartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_13041conv1d_2_13043*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????V*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_127292"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_126552 
MaxPooling1D_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_129542#
!dropout_1/StatefulPartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_13048conv1d_3_13050*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????"/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_127592"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_126702 
MaxPooling1D_3/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_129212#
!dropout_2/StatefulPartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_13055conv1d_4_13057*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_127892"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_128012
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_13061dense_1_13063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_128142!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_13066dense_2_13068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_128312!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_13276

inputs
unknown:
	unknown_0:
	unknown_1:V
	unknown_2:V
	unknown_3:V/
	unknown_4:/
	unknown_5:/#
	unknown_6:#
	unknown_7:	?y
	unknown_8:y
	unknown_9:y!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_128382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_13574

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????$V* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_127412
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????$V2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????$V:S O
+
_output_shapes
:?????????$V
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_12789

inputsA
+conv1d_expanddims_1_readvariableop_resource:/#-
biasadd_readvariableop_resource:#
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????/2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:/#*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:/#2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????#*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:#*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????#2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????#2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_13305

inputs
unknown:
	unknown_0:
	unknown_1:V
	unknown_2:V
	unknown_3:V/
	unknown_4:/
	unknown_5:/#
	unknown_6:#
	unknown_7:	?y
	unknown_8:y
	unknown_9:y!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_130722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_12865
input_1
unknown:
	unknown_0:
	unknown_1:V
	unknown_2:V
	unknown_3:V/
	unknown_4:/
	unknown_5:/#
	unknown_6:#
	unknown_7:	?y
	unknown_8:y
	unknown_9:y!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_128382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_12771

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????/2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????/2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????/:S O
+
_output_shapes
:?????????/
 
_user_specified_nameinputs
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_12831

inputs0
matmul_readvariableop_resource:y!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:y!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????y: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????y
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_13532

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_1_layer_call_fn_13693

inputs
unknown:	?y
	unknown_0:y
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????y*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_128142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????y2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????;
Dense_20
StatefulPartitionedCall:0?????????!tensorflow/serving/predict:ا
?t
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?o
_tf_keras_network?o{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1836852444307216, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 86, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.15025703514120636, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 47, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.18147391748885905, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 35, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 121, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}, "shared_object_id": 26, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1836852444307216, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 86, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.15025703514120636, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 47, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.18147391748885905, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 35, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 121, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 25}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0014400000218302011, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 28, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
?
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1836852444307216, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 5}
?


#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 86, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 28}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 147, 28]}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.15025703514120636, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 10}
?


1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 47, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 86}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 86]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "MaxPooling1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}}
?
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.18147391748885905, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}
?


?kernel
@bias
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1D_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 35, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 47}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 47]}}
?
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 36}}
?	

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 121, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 525}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 525]}}
?	

Okernel
Pbias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 121}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 121]}}
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?1m?2m??m?@m?Im?Jm?Om?Pm?v?v?#v?$v?1v?2v??v?@v?Iv?Jv?Ov?Pv?"
	optimizer
v
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11"
trackable_list_wrapper
?
trainable_variables
Zmetrics
regularization_losses
[non_trainable_variables
	variables

\layers
]layer_metrics
^layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#2Conv1D_1/kernel
:2Conv1D_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
_metrics
	variables
`non_trainable_variables
regularization_losses

alayers
blayer_metrics
clayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
dmetrics
	variables
enon_trainable_variables
regularization_losses

flayers
glayer_metrics
hlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
imetrics
 	variables
jnon_trainable_variables
!regularization_losses

klayers
llayer_metrics
mlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#V2Conv1D_2/kernel
:V2Conv1D_2/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%trainable_variables
nmetrics
&	variables
onon_trainable_variables
'regularization_losses

players
qlayer_metrics
rlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables
smetrics
*	variables
tnon_trainable_variables
+regularization_losses

ulayers
vlayer_metrics
wlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
xmetrics
.	variables
ynon_trainable_variables
/regularization_losses

zlayers
{layer_metrics
|layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#V/2Conv1D_3/kernel
:/2Conv1D_3/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3trainable_variables
}metrics
4	variables
~non_trainable_variables
5regularization_losses

layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7trainable_variables
?metrics
8	variables
?non_trainable_variables
9regularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;trainable_variables
?metrics
<	variables
?non_trainable_variables
=regularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#/#2Conv1D_4/kernel
:#2Conv1D_4/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Atrainable_variables
?metrics
B	variables
?non_trainable_variables
Cregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Etrainable_variables
?metrics
F	variables
?non_trainable_variables
Gregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?y2Dense_1/kernel
:y2Dense_1/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ktrainable_variables
?metrics
L	variables
?non_trainable_variables
Mregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :y!2Dense_2/kernel
:!2Dense_2/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qtrainable_variables
?metrics
R	variables
?non_trainable_variables
Sregularization_losses
?layers
?layer_metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 39}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(2Adam/Conv1D_1/kernel/m
 :2Adam/Conv1D_1/bias/m
*:(V2Adam/Conv1D_2/kernel/m
 :V2Adam/Conv1D_2/bias/m
*:(V/2Adam/Conv1D_3/kernel/m
 :/2Adam/Conv1D_3/bias/m
*:(/#2Adam/Conv1D_4/kernel/m
 :#2Adam/Conv1D_4/bias/m
&:$	?y2Adam/Dense_1/kernel/m
:y2Adam/Dense_1/bias/m
%:#y!2Adam/Dense_2/kernel/m
:!2Adam/Dense_2/bias/m
*:(2Adam/Conv1D_1/kernel/v
 :2Adam/Conv1D_1/bias/v
*:(V2Adam/Conv1D_2/kernel/v
 :V2Adam/Conv1D_2/bias/v
*:(V/2Adam/Conv1D_3/kernel/v
 :/2Adam/Conv1D_3/bias/v
*:(/#2Adam/Conv1D_4/kernel/v
 :#2Adam/Conv1D_4/bias/v
&:$	?y2Adam/Dense_1/kernel/v
:y2Adam/Dense_1/bias/v
%:#y!2Adam/Dense_2/kernel/v
:!2Adam/Dense_2/bias/v
?2?
%__inference_model_layer_call_fn_12865
%__inference_model_layer_call_fn_13276
%__inference_model_layer_call_fn_13305
%__inference_model_layer_call_fn_13128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_13388
@__inference_model_layer_call_and_return_conditional_losses_13492
@__inference_model_layer_call_and_return_conditional_losses_13169
@__inference_model_layer_call_and_return_conditional_losses_13210?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_12631?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
(__inference_Conv1D_1_layer_call_fn_13501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_13517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_1_layer_call_fn_12646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12640?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
'__inference_dropout_layer_call_fn_13522
'__inference_dropout_layer_call_fn_13527?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_13532
B__inference_dropout_layer_call_and_return_conditional_losses_13544?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv1D_2_layer_call_fn_13553?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_13569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_2_layer_call_fn_12661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12655?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
)__inference_dropout_1_layer_call_fn_13574
)__inference_dropout_1_layer_call_fn_13579?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13584
D__inference_dropout_1_layer_call_and_return_conditional_losses_13596?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv1D_3_layer_call_fn_13605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_13621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_MaxPooling1D_3_layer_call_fn_12676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12670?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
)__inference_dropout_2_layer_call_fn_13626
)__inference_dropout_2_layer_call_fn_13631?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13636
D__inference_dropout_2_layer_call_and_return_conditional_losses_13648?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_Conv1D_4_layer_call_fn_13657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_13673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_13678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_13684?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_1_layer_call_fn_13693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_2_layer_call_fn_13713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_2_layer_call_and_return_conditional_losses_13724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_13247input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_13517f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
(__inference_Conv1D_1_layer_call_fn_13501Y4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_13569f#$4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????V
? ?
(__inference_Conv1D_2_layer_call_fn_13553Y#$4?1
*?'
%?"
inputs??????????
? "???????????V?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_13621d123?0
)?&
$?!
inputs?????????$V
? ")?&
?
0?????????"/
? ?
(__inference_Conv1D_3_layer_call_fn_13605W123?0
)?&
$?!
inputs?????????$V
? "??????????"/?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_13673d?@3?0
)?&
$?!
inputs?????????/
? ")?&
?
0?????????#
? ?
(__inference_Conv1D_4_layer_call_fn_13657W?@3?0
)?&
$?!
inputs?????????/
? "??????????#?
B__inference_Dense_1_layer_call_and_return_conditional_losses_13704]IJ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????y
? {
'__inference_Dense_1_layer_call_fn_13693PIJ0?-
&?#
!?
inputs??????????
? "??????????y?
B__inference_Dense_2_layer_call_and_return_conditional_losses_13724\OP/?,
%?"
 ?
inputs?????????y
? "%?"
?
0?????????!
? z
'__inference_Dense_2_layer_call_fn_13713OOP/?,
%?"
 ?
inputs?????????y
? "??????????!?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_12640?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_1_layer_call_fn_12646wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_12655?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_2_layer_call_fn_12661wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_12670?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_3_layer_call_fn_12676wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
 __inference__wrapped_model_12631x#$12?@IJOP5?2
+?(
&?#
input_1??????????
? "1?.
,
Dense_2!?
Dense_2?????????!?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13584d7?4
-?*
$?!
inputs?????????$V
p 
? ")?&
?
0?????????$V
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_13596d7?4
-?*
$?!
inputs?????????$V
p
? ")?&
?
0?????????$V
? ?
)__inference_dropout_1_layer_call_fn_13574W7?4
-?*
$?!
inputs?????????$V
p 
? "??????????$V?
)__inference_dropout_1_layer_call_fn_13579W7?4
-?*
$?!
inputs?????????$V
p
? "??????????$V?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13636d7?4
-?*
$?!
inputs?????????/
p 
? ")?&
?
0?????????/
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_13648d7?4
-?*
$?!
inputs?????????/
p
? ")?&
?
0?????????/
? ?
)__inference_dropout_2_layer_call_fn_13626W7?4
-?*
$?!
inputs?????????/
p 
? "??????????/?
)__inference_dropout_2_layer_call_fn_13631W7?4
-?*
$?!
inputs?????????/
p
? "??????????/?
B__inference_dropout_layer_call_and_return_conditional_losses_13532f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_13544f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
'__inference_dropout_layer_call_fn_13522Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
'__inference_dropout_layer_call_fn_13527Y8?5
.?+
%?"
inputs??????????
p
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_13684]3?0
)?&
$?!
inputs?????????#
? "&?#
?
0??????????
? {
'__inference_flatten_layer_call_fn_13678P3?0
)?&
$?!
inputs?????????#
? "????????????
@__inference_model_layer_call_and_return_conditional_losses_13169t#$12?@IJOP=?:
3?0
&?#
input_1??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_13210t#$12?@IJOP=?:
3?0
&?#
input_1??????????
p

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_13388s#$12?@IJOP<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_13492s#$12?@IJOP<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????!
? ?
%__inference_model_layer_call_fn_12865g#$12?@IJOP=?:
3?0
&?#
input_1??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_13128g#$12?@IJOP=?:
3?0
&?#
input_1??????????
p

 
? "??????????!?
%__inference_model_layer_call_fn_13276f#$12?@IJOP<?9
2?/
%?"
inputs??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_13305f#$12?@IJOP<?9
2?/
%?"
inputs??????????
p

 
? "??????????!?
#__inference_signature_wrapper_13247?#$12?@IJOP@?=
? 
6?3
1
input_1&?#
input_1??????????"1?.
,
Dense_2!?
Dense_2?????????!