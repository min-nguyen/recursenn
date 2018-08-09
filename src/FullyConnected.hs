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
     RecordWildCards,
     ExistentialQuantification #-}

module FullyConnected where

import Utils
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace


---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Fully Connected NN                      | | ---
  --- ‾------------------------------------------------------------------‾---


data Layer k where
    Layer       :: Weights -> Biases -> (Activation, Activation') -> k -> Layer k
    InputLayer  :: Layer k 
    deriving Show

instance Functor (Layer) where
    fmap eval (Layer weights biases activate k)      = Layer weights biases activate (eval k) 
    fmap eval (InputLayer )                          = InputLayer 

data BackPropData       = BackPropData  { 
                                         inputStack     :: [Inputs],
                                         desiredOutput  :: DesiredOutput, 
                                         outerDeltas    :: Deltas, 
                                         outerWeights   :: Weights
                                        }

type Weights            = [[Double]]
type Biases             = [Double]
type Inputs             = [Double]
type Outputs            = [Double]
type Activation         =  Double  ->  Double
type Activation'        =  Double  ->  Double
type LossFunction       = [Double] -> [Double] -> Double
type DesiredOutput      = [Double]
type FinalOutput        = [Double]
type Deltas             = [Double]


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
compDelta derivActivation inputs outputs (BackPropData _  desiredOutput outerDeltas outerWeights)   
    =   let z = map inverseSigmoid inputs
        in  case outerDeltas of [] -> elemul (zipWith (-) outputs desiredOutput) (map derivActivation z)
                                _  -> elemul (mvmul (transpose outerWeights) outerDeltas) (map derivActivation z)

forward :: Weights -> Biases -> Activation -> ([Inputs] -> [Inputs]) -> ([Inputs] -> [Inputs])
forward weights biases activate k 
    = (\inputs -> (map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

backward :: Weights -> Biases -> Inputs  -> BackPropData -> (Weights, Biases)
backward weights biases inputs (BackPropData {outerDeltas = updatedDeltas, ..} )
    = let learningRate = 0.2
          inputsDeltasWeights = map (zip3 inputs updatedDeltas) weights
          updatedWeights = [[ w - learningRate*d*i  |  (i, d, w) <- idw_vec ] | idw_vec <- inputsDeltasWeights]                                                  
          updatedBiases  = zipWith (-) biases (map (learningRate *) updatedDeltas)
      in (updatedWeights, updatedBiases)

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                    Running And Constructing NNs                | | ---
  --- ‾------------------------------------------------------------------‾---

train :: Fix Layer -> LossFunction -> Inputs -> DesiredOutput -> Fix Layer 
train neuralnet lossfunction sample desiredoutput 
    = trace (show $ head inputStack) $ 
        ella coalgx coalgy $ (nn, BackPropData inputStack desiredoutput [] [[]] )
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

runFullyConnected = print $ show $ train example loss [1.0, 2.0, 0.2] [-26.0, 5.0, 3.0]