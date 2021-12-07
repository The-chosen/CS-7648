# 1st Training
echo '==========================================================================================' 
echo '==========================================================================================' 
echo '>> 1st Training' 
echo 'Human' 
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 1 


echo '==========================================================================================' 
echo 'rl' 

## Baseline1(rl  

### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 1 


echo '==========================================================================================' 
echo 'safe' 

## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 1 

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 1 



echo '==========================================================================================' 














# 2st Training
echo '==========================================================================================' 
echo '>> 2st Training' 
echo 'Human' 
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 2

echo '==========================================================================================' 
echo 'rl' 
## Baseline1(rl  
### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 2


echo '==========================================================================================' 
echo 'safe' 
## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 2

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 2



echo '==========================================================================================' 
















# 3st Training
echo '==========================================================================================' 
echo '>> 3st Training' 
echo 'Human' 
## Human
### env-default
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode rl --isHumanBuffer True --explore rnd --env cross --trainNum 3 


echo '==========================================================================================' 
echo 'rl' 
## Baseline1(rl  
### env-default
python train.py --display none  --no-qp --mode rl --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode rl --explore rnd --env cross --trainNum 3


echo '==========================================================================================' 
echo 'safe' 

## Safe
### env-default
python train.py --display none  --no-qp --mode safe --explore rnd --trainNum 3

### env-cross
python train.py --display none  --no-qp --mode safe --explore rnd --env cross --trainNum 3


echo '==========================================================================================' 