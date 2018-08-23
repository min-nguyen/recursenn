{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     TemplateHaskell, RankNTypes, DeriveFoldable,
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

module FullyConnectedx where

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
import Control.Lens
import Data.Maybe
import Control.Lens.Tuple
import Types
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Fully Connected NN                      | | ---
  --- ‾------------------------------------------------------------------‾---


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

data FullyConnected f b k = 
        FullyConnected   { 
                  _weights      :: Weights,
                  _biases       :: Biases,
                  _activation   :: Activation,
                  _activation'  :: Activation',
                  _nextLayer    :: k
                } 
    |   InputLayer  deriving (Show, Functor, Foldable, Traversable)
makeLenses ''FullyConnected

data BackPropData   = 
    BackPropData   { 
        _inputStack     :: [Inputs],
        _desiredOutput  :: DesiredOutput, 
        _outerDeltas    :: Deltas, 
        _outerWeights   :: Weights
                    } deriving (Show)
makeLenses ''BackPropData

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

type FP = ([Inputs] -> [Inputs])
type BP = BackPropData

alg :: FullyConnected FP BP (Fix (FullyConnected FP BP), ([Inputs] -> [Inputs]) ) -> (Fix (FullyConnected FP BP), ([Inputs] -> [Inputs]))
alg InputLayer 
    =  (Fx InputLayer, id)
alg layer   
    =  (Fx (layer & nextLayer %~ fst), forward layer)

coalg :: (Fix (FullyConnected FP BP), BackPropData) -> FullyConnected FP BP  (Fix (FullyConnected FP BP), BackPropData)
coalg (Fx InputLayer, output)
    =  InputLayer  
coalg (Fx layer, bp)
    =   let delta = compDelta (fromJust $ layer ^? activation') bp
            bp' =  bp & inputStack  %~ tail 
                      & outerDeltas .~ delta
            (newWeights, newBiases) = (backward (layer ^. weights) (layer ^. biases) bp')
        in layer & weights   .~ newWeights 
                 & biases    .~ newBiases 
                 & nextLayer %~ \x -> (x, bp' & outerWeights .~ (layer ^. weights))


instance Layer FullyConnected FP BP where
    runForward = alg
    runBackward = coalg

-- ---- |‾| -------------------------------------------------------------- |‾| ----
--  --- | |                    Forward & Back Propagation                  | | ---
--   --- ‾------------------------------------------------------------------‾---

compDelta ::  Activation' -> BackPropData -> Deltas 
compDelta derivActivation (BackPropData (outputs:inputs:xs) desiredOutput outerDeltas outerWeights)   
    =   let z = map inverseSigmoid inputs
        in  case outerDeltas of [] -> elemul (zipWith (-) outputs desiredOutput) (map derivActivation z)
                                _  -> elemul (mvmul (transpose outerWeights) outerDeltas) (map derivActivation z)

forward :: FullyConnected FP BP (Fix (FullyConnected FP BP), ([Inputs] -> [Inputs]) ) -> ([Inputs] -> [Inputs])
forward (FullyConnected weights biases activate activate' (innerLayer, k) )
    = (\inputs -> (map activate 
        ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

backward :: Weights -> Biases  -> BackPropData -> (Weights, Biases)
backward weights biases BackPropData {_inputStack = (inputs:xs), _outerDeltas = updatedDeltas, ..}
    = let learningRate = 0.2
          inputsDeltasWeights = map (zip3 inputs updatedDeltas) weights
          updatedWeights = [[ w - learningRate*d*i  |  (i, d, w) <- idw_vec ] | idw_vec <- inputsDeltasWeights]                                                  
          updatedBiases  = zipWith (-) biases (map (learningRate *) updatedDeltas)
      in (updatedWeights, updatedBiases)

-- -- ---- |‾| -------------------------------------------------------------- |‾| ----
-- --  --- | |                    Running And Constructing NNs                | | ---
-- --   --- ‾------------------------------------------------------------------‾---

train :: Fix (FullyConnected FP BP) -> Inputs -> DesiredOutput -> Fix (FullyConnected FP BP) 
train neuralnet sample desiredoutput 
    =  meta alg (\(nn, diff_fun) -> (nn, BackPropData (diff_fun [sample]) desiredoutput [] [[]] )) coalg $ neuralnet

trains :: Fix (FullyConnected FP BP) -> [Inputs] -> [DesiredOutput] -> Fix (FullyConnected FP BP)
trains neuralnet samples desiredoutputs  
    = foldr (\(sample, desiredoutput) nn -> train nn sample desiredoutput) neuralnet (zip samples desiredoutputs)

construct :: [(Weights, Biases, Activation, Activation')] -> Fix (FullyConnected FP BP)
construct (x:xs) = Fx (FullyConnected weights biases activation activation' (construct (xs)))
            where (weights, biases, activation, activation') = x
construct []       = Fx InputLayer


example =  (Fx ( FullyConnected [[3.0,6.0,2.0],[2.0,1.0,7.0],[6.0,5.0,2.0]] [0, 0, 0] sigmoid sigmoid'
            (Fx ( FullyConnected [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] [0, 0, 0] sigmoid sigmoid'
             (Fx   InputLayer ) ) ) ) )

runFullyConnected = print $ show $ train example [1.0, 2.0, 0.2] [-26.0, 5.0, 3.0]