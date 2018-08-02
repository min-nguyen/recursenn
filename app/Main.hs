{-# LANGUAGE
  
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     ScopedTypeVariables,
     MultiParamTypeClasses,
     FlexibleContexts,
     TypeFamilies,
     GADTs,
     DataKinds,
     KindSignatures #-}

module Main where

import Lib
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
import qualified V as V
import V (Vector((:-)))
import Debug.Trace


sigmoid :: Double -> Double
sigmoid lx = 1.0 / (1.0 + exp (negate lx))


loss :: Fractional a => [a] -> [a] -> a
loss output desired_output 
    = (1/(fromIntegral $ length output)) * (sum $ map ((\x -> x*x) . (abs)) (zipWith (-) output desired_output))

sigmoid' :: Double -> Double
sigmoid' x = let sig = (sigmoid x) in sig * (1.0 - sig)
    
----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------    
data Layer k where
    Layer :: Weights -> Biases -> (Activation, Activation') -> k -> Layer k
    InputLayer :: Layer k 
    deriving Show

instance Functor (Layer) where
    fmap eval (Layer weights biases activate k)      = Layer weights biases activate (eval k) 
    fmap eval (InputLayer )                          = InputLayer 

forward :: Weights -> Biases -> Activation -> ([Inputs] -> [Inputs]) -> ([Inputs] -> [Inputs])
forward weights biases activate k 
    = (\inputs -> (map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

alg :: Layer (Fix Layer, ([Inputs] -> [Inputs]) ) -> (Fix Layer, ([Inputs] -> [Inputs]))
alg (Layer weights biases (activate, activate') (innerLayer, forwardPass) )   
    =  (Fx (Layer weights biases (activate, activate') innerLayer ) , updated_inputs )
        where updated_inputs = (forward weights biases activate forwardPass)
alg (InputLayer )                     
    =  (Fx InputLayer, id)

compDelta :: Weights -> Deltas -> Activation' -> [Double] -> [Double] -> [Double] -> [Double]
compDelta outerWeights outerDeltas derivActivation inputs outputs desiredOutput  
    = case outerDeltas of  [] -> elemul (map (\x -> x*(x-1)) outputs) (zipWith (-) outputs desiredOutput)
                           _  -> elemul (mvmul (transpose outerWeights) outerDeltas) (map derivActivation inputs)

backward :: Weights -> Biases -> Activation' -> [Double] -> [Double] -> [Double] -> [Double] -> Deltas -> Weights -> (Deltas, Weights)
backward weights biases activate' input output finalOutput desiredOutput outerDelta outerWeights
    = let learningRate = 0.2 
          deltas = compDelta outerWeights outerDelta activate' input output desiredOutput
          newWeights = [[ w - learningRate*d*i  |  (i, w) <- zip input weightvec ] | d <- deltas, weightvec <- weights]                                                  
      in (deltas, newWeights)

coalg :: (Fix Layer, ([Inputs], FinalOutput, DesiredOutput, Deltas, Weights)) -> Layer  (Fix Layer, ([Inputs], FinalOutput, DesiredOutput, Deltas, Weights)) 
coalg (Fx (Layer weights biases (activate, activate') innerLayer), 
            ((output:input:xs), finalOutput, desiredOutput, outerDelta, outerWeights))
    =   Layer newWeights biases (activate, activate') 
                (innerLayer, ((input:xs), finalOutput, desiredOutput, delta, weights))
        where (delta, newWeights) = 
                (backward weights biases activate' input output finalOutput desiredOutput outerDelta outerWeights)
coalg (Fx InputLayer, output)
    =  InputLayer 

train :: Fix Layer -> LossFunction -> Inputs -> DesiredOutput -> Fix Layer 
train neuralnet lossfunction sample desiredoutput 
    = trace (show $ head activation_values) $ ana coalg $ (nn, (activation_values, head activation_values, desiredoutput, [], [[]]) )
        where 
            (nn, diff_fun)      = cata alg neuralnet
            activation_values   = diff_fun [sample]

trains :: Fix Layer -> LossFunction -> [Inputs] -> [DesiredOutput] -> Fix Layer
trains neuralnet lossfunction samples desiredoutputs  
    = foldr (\(sample, desiredoutput) nn -> train nn lossfunction sample desiredoutput) neuralnet (zip samples desiredoutputs)

construct :: [(Weights, Biases, Activation, Activation')] -> Fix Layer
construct (x:xs) = Fx (Layer weights biases (activation, activation') (construct (xs)))
    where (weights, biases, activation, activation') = x
construct []       = Fx InputLayer


example =  (Fx ( Layer [[3.0,6.0,2.0],[2.0,1.0,7.0],[6.0,5.0,2.0]] [0, 0, 0] (sigmoid, sigmoid')
            (Fx ( Layer [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] [0, 0, 0] (sigmoid, sigmoid')
             (Fx   InputLayer ) ) ) ) )

main = print $ show $ train example loss [1.0, 2.0, 0.2] [-26.0, 5.0, 3.0]

newtype Fox f g = Fo (f (Fox g f))

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

-- newtype Fox f o i = Fo (f o i (Fox f V.Nat o) )

-- data Layer' (o::V.Nat) (i::V.Nat) (k:: * -> V.Nat -> *) where
--     Layer' :: V.Vector (V.Vector Double i ) o -> V.Vector Double i -> Activation -> k (V.Nat ) o -> Layer' o i k
--     InputLayer' :: Layer' o i k 


-- data Layer' (o::V.Nat) (i::V.Nat) k where
--     Layer' :: V.Vector (V.Vector Double i ) o  -> V.Vector Double i  -> Activation -> k -> Layer' o i k
--     InputLayer' :: Layer' o i k 
--     deriving Show
    
-- instance Functor (Layer' o i) where
--     fmap eval (Layer' weights biases activate k)      = Layer' weights biases activate (eval k) 
--     fmap eval (InputLayer' )                          = InputLayer' 

-- forward' :: Fractional a => V.Vector (V.Vector a i) o -> V.Vector a i -> (a -> a) -> ([[a]] -> [[a]]) -> ([[a]] -> [[a]]) 
-- forward' weights biases activate k 
--     = (\inputs ->  ((map activate (zipWith (+) 
--             ((V.toList $ V.map 
--                 ((sum)  . (zipWithPadding (*) (head inputs)) . (V.toList)) weights)) (V.toList biases))):inputs) ) . k

-- backward' :: (V.SingRep o, V.SingRep i, Fractional a) => V.Vector (V.Vector a i) o -> V.Vector a i -> [a] -> [a] -> V.Vector (V.Vector a i) o 
-- backward' weights biases input final_output 
--     = let   list_weights = (V.toList $ ((V.map V.toList weights)) )
--             learning_rate = 1
--             desired_output = 3
--             error = (sum final_output) / (fromIntegral $ length final_output)
--             new_list_weights = transpose $ map (zipWith (+) (map (\xi -> learning_rate * xi * (desired_output - error)) input )) (transpose list_weights)
--       in    V.unsafeFromList' $ map V.unsafeFromList' $ new_list_weights

-- alg' :: Layer' o i (Fox (Layer' j o) (Layer' o i ), ([Inputs] -> [Inputs]) ) 
--         -> (Fox (Layer' o i) (Layer' j o ), ([Inputs] -> [Inputs]))
-- alg' (Layer' weights biases activate (innerLayer, forwardPass) )   
--     =  (Fo (Layer' weights biases activate innerLayer ) , (forward' weights biases activate forwardPass) )
-- alg' (InputLayer' )                     
--     =  (Fo InputLayer', id )

-- coalg' :: (V.SingRep o, V.SingRep i) => (Fox (Layer' o i) (Layer' j o), [Inputs]) 
--         -> Layer' o i (Fox (Layer' j o) (Layer' l j), [Inputs]) 
-- coalg' (Fo (Layer' weights biases activate innerLayer), (x:y:ys))
--     =  Layer' (backward' weights biases y x) biases activate (innerLayer, (x:ys))
-- coalg' (Fo InputLayer', output)      
--     =  InputLayer' 

-- train' :: (V.SingRep o, V.SingRep i) => Fox (Layer' o i) (Layer' j o) -> LossFunction 
--         -> Inputs -> (Fox (Layer' o i) (Layer' j o)) 
-- train' neuralnet lossfunction sample = ana' $ (nn,  diff_fun [sample])
--   where 
--     (nn, diff_fun) = cata' neuralnet

-- cata' ::  Fox (Layer' o0 i0) (Layer' i0 o0) -> (Fox (Layer' o0 i0) (Layer' i0 o0), [Inputs] -> [Inputs])
-- cata' = alg' . fmap (cata'') . unFox

-- cata'' :: Fox (Layer' o1 i2) (Layer' i2 o1) -> (Fox (Layer' o1 i2) (Layer' i2 o1), [Inputs] -> [Inputs])
-- cata'' = alg' . fmap (cata') . unFox

-- ana' :: (V.SingRep o, V.SingRep i) => (Fox (Layer' o i) (Layer' j o), [Inputs]) -> Fox (Layer' o i) (Layer' j o)
-- ana' = Fo . fmap (ana'') . coalg'

-- ana'' :: (V.SingRep o, V.SingRep i) => (Fox (Layer' o i) (Layer' j o), [Inputs]) -> Fox (Layer' o i) (Layer' j o)
-- ana'' = Fo . fmap (ana') . coalg'

-- unFox :: Fox f g -> f (Fox g f)
-- unFox (Fo x) = x

-- example' =  (Fx ( Layer' ((3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--              (Fx ( Layer' ((3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--               (Fx   InputLayer' ) ) ) ) )
