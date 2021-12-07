# 1st Training
print('==========================================================================================')
print('==========================================================================================')
print('>> 1st Training')
print('Human')
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 1 


print('==========================================================================================')
print('rl')

## Baseline1(rl) 

### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 1 


print('==========================================================================================')
print('safe')

## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 1 



print('==========================================================================================')














# 2st Training
print('==========================================================================================')
print('>> 1st Training')
print('Human')
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 2

print('==========================================================================================')
print('rl')
## Baseline1(rl) 
### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 2


print('==========================================================================================')
print('safe')
## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 2



print('==========================================================================================')
















# 3st Training
print('==========================================================================================')
print('>> 1st Training')
print('Human')
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 3 


print('==========================================================================================')
print('rl')
## Baseline1(rl) 
### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 3


print('==========================================================================================')
print('safe')

## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 3


print('==========================================================================================')