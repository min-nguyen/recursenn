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
     KindSignatures,
     RecordWildCards #-}

module Main where

import Lib
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
import qualified Vector as Vector
import Vector (Vector((:-)))
import Debug.Trace

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

algx :: Layer (Fix Layer, ([Inputs] -> [Inputs]) ) -> Layer (Fix Layer)
algx (Layer weights biases (activate, activate') (innerLayer, forwardPass) )   
    =  Layer weights biases (activate, activate') innerLayer
algx InputLayer 
    =  InputLayer

algy :: Layer (Fix Layer, ([Inputs] -> [Inputs]) ) -> ([Inputs] -> [Inputs])
algy (Layer weights biases (activate, _) (innerLayer, forwardPass) )
    = forward weights biases activate forwardPass   
algy InputLayer
    = id

coalgx :: (Fix Layer, BackPropData) -> (BackPropData -> Layer  (Fix Layer, BackPropData) )
coalgx (Fx (Layer weights biases (activate, activate') innerLayer), (BackPropData { inputStack = (output:input:xs), .. }))
    =  \backPropData -> let (newWeights, newBiases) = (backward weights biases input backPropData)
                        in Layer newWeights newBiases (activate, activate') (innerLayer, backPropData {outerWeights = weights})
coalgx (Fx InputLayer, output)
    =  \_ -> InputLayer 

coalgy :: (Fix Layer, BackPropData) -> BackPropData
coalgy (Fx (Layer weights biases (activate, activate') innerLayer), backPropData)
    =   let BackPropData { inputStack = (outputs:inputs:xs), .. } = backPropData
            delta = compDelta activate' inputs outputs backPropData
        in  backPropData { inputStack = (inputs:xs), outerDeltas = delta }
coalgy (Fx InputLayer, backPropData)
    =   backPropData

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                    Forward & Back Propagation                  | | ---
  --- ‾------------------------------------------------------------------‾---

compDelta ::  Activation' -> Inputs -> Outputs -> BackPropData -> Deltas 
compDelta derivActivation inputs outputs (BackPropData _ finalOutput desiredOutput outerDeltas outerWeights)   
    = case outerDeltas of  [] -> elemul (map (\x -> x*(x-1)) outputs) (zipWith (-) outputs desiredOutput)
                           _  -> elemul (mvmul (transpose outerWeights) outerDeltas) (map derivActivation inputs)

forward :: Weights -> Biases -> Activation -> ([Inputs] -> [Inputs]) -> ([Inputs] -> [Inputs])
forward weights biases activate k 
    = (\inputs -> (map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

backward :: Weights -> Biases -> Inputs  -> BackPropData -> (Weights, Biases)
backward weights biases inputs (BackPropData {outerDeltas = updatedDeltas, ..} )
    = let learningRate = 0.2
          inputsDeltasWeights = map (zip3 inputs updatedDeltas) weights
          updatedWeights = [[ w - learningRate*d*i  |  (i, w, d) <- idw_vec ] | idw_vec <- inputsDeltasWeights]                                                  
          updatedBiases  = map (learningRate *) updatedDeltas
      in (updatedWeights, updatedBiases)

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                    Running And Constructing NNs                | | ---
  --- ‾------------------------------------------------------------------‾---

train :: Fix Layer -> LossFunction -> Inputs -> DesiredOutput -> Fix Layer 
train neuralnet lossfunction sample desiredoutput 
    = trace (show $ head inputStack) $ 
        ella coalgx coalgy $ (nn, BackPropData inputStack (head inputStack) desiredoutput [] [[]] )
            where 
                (nn, diff_fun)      = doggo algx algy neuralnet
                inputStack   = diff_fun [sample]

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





newtype Fox f g = Fox (f (Fox g f))



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
