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
import Control.Lens
import Control.Monad
import Control.Monad.Trans.Class
import Data.Maybe
import Control.Lens.Tuple
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

data Layer k = 
        Layer   { 
                  _weights      :: Weights,
                  _biases       :: Biases,
                  _activation   :: Activation,
                  _activation'  :: Activation',
                  _nextLayer    :: k
                } 
    |   InputLayer  deriving (Show, Functor, Foldable, Traversable)
makeLenses ''Layer

data PropData   = 
    PropData   { 
        _forwardPass    :: ([Inputs] -> [Inputs]),
        _inputStack     :: [Inputs],
        _desiredOutput  :: DesiredOutput, 
        _outerDeltas    :: Deltas, 
        _outerWeights   :: Weights
                    } deriving (Show)
makeLenses ''PropData

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

alg :: Layer (Fix Layer, PropData ) -> (Fix Layer, PropData)
alg InputLayer 
    =  (Fx InputLayer, forward (InputLayer))
alg (Layer weights biases activate activate' (innerLayer, propData) )   
    =  (Fx (Layer weights biases activate activate' innerLayer), (forward (Layer weights biases activate activate' (innerLayer, propData) )) )

coalg :: (Fix Layer, PropData) -> Layer  (Fix Layer, PropData)
coalg (Fx InputLayer, output)
    =  InputLayer  
coalg (Fx layer, bp)
    =   let delta = compDelta (fromJust $ layer ^? activation') bp
            bp' =  bp & inputStack  %~ tail 
                      & outerDeltas .~ delta
            (newWeights, newBiases) = (backward (layer ^. weights) (layer ^. biases) bp')
        in 
            layer & weights   .~ newWeights 
                 & biases    .~ newBiases 
                 & nextLayer %~ \x -> (x, bp' & outerWeights .~ (layer ^. weights))


-- ---- |‾| -------------------------------------------------------------- |‾| ----
--  --- | |                    Forward & Back Propagation                  | | ---
--   --- ‾------------------------------------------------------------------‾---

compDelta ::  Activation' -> PropData -> Deltas 
compDelta derivActivation (PropData fp (outputs:inputs:xs) desiredOutput outerDeltas outerWeights)   
    =   let sigmoid'_z = (map derivActivation) (map inverseSigmoid inputs) 
        in  case outerDeltas of [] -> let cost = elemul (zipWith (-) outputs desiredOutput) sigmoid'_z -- we're dealing with the last layer
                                      in  writeResult (((\z -> showFullPrecision $ abs $ read $ formatFloatN (z) 8) $ head (zipWith (-) outputs desiredOutput))) cost
                                _  -> elemul (mvmul (transpose outerWeights) outerDeltas) sigmoid'_z -- we're dealing with any other layer than the last

trydelta = compDelta sigmoid' (PropData id [[0.975377100, 0.895021978, 0.956074004], [0.668187772, 0.937026644, 0.2689414214]] [0.0, 1.0, 0.0] [] [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] )

forward :: Layer (Fix Layer, PropData) -> PropData
forward (Layer weights biases activate activate' (innerLayer, propData) )
    = propData & forwardPass .~ f
      where fp = propData ^. forwardPass
            f = ((\inputs -> (map activate 
                    (zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases)):inputs) . fp)
forward (InputLayer) = PropData f [[]] [] [] [[]]
    where f = (\(inputs:_) -> (inputs:inputs:[]))

backward :: Weights -> Biases  -> PropData -> (Weights, Biases)
backward weights biases PropData {_inputStack = (inputs:prev_inputs:xs), _outerDeltas = updatedDeltas, ..}
    = let learningRate = 2
          weightGradient = transpose $ map2 (\x -> learningRate * x) (outerproduct prev_inputs updatedDeltas)
          updatedWeights = elesubm weights weightGradient --[[ w - learningRate*d*i  |  (i, d, w) <- idw_vec ] | idw_vec <- inputsDeltasWeights]                                                  
          updatedBiases  = zipWith (-) biases (map (learningRate *) updatedDeltas)
      in  --trace ("Updated deltas: " ++ show updatedDeltas ++ " weights : " ++ show weightGradient) 
          (updatedWeights, updatedBiases)

-- ---- |‾| -------------------------------------------------------------- |‾| ----
--  --- | |                    Running And Constructing NNs                | | ---
--   --- ‾------------------------------------------------------------------‾---

train :: Fix Layer -> Inputs -> DesiredOutput -> Fix Layer
train neural_net sample desired_output
    = meta alg h coalg $ neural_net
      where h :: (Fix Layer, PropData) -> (Fix Layer, PropData)
            h (nn, pd) = (nn, PropData id ((pd ^. forwardPass) [sample]) desired_output [] [[]])


trains :: Fix Layer -> [Inputs] -> [DesiredOutput] -> Fix Layer
trains neuralnet samples desiredoutputs  
    = foldr (\(sample, desiredoutput) nn -> 
                  let updatedNetwork = train nn sample desiredoutput
                  in   updatedNetwork) neuralnet (zip samples desiredoutputs)

deforest :: Fix Layer -> Fix Layer 
deforest nn = meta deforestCata g deforestAna $  nn
        where g :: (Fix Layer, PropData) -> Fix Layer 
              g (fx_layer, bp) = fx_layer

deforestCata :: Layer (Fix Layer, PropData) -> (Fix Layer, PropData)
deforestCata (Layer weights biases activate activate' (innerLayer, k)) 
        = (Fx (Layer weights biases activate activate' innerLayer), k)
deforestCata (InputLayer) 
        = (Fx InputLayer, PropData id [] [] [] [[]])

deforestAna :: (Fix Layer) -> Layer (Fix Layer)
deforestAna (Fx  (Layer weights biases activate activate' innerLayer))
        = Layer weights biases activate activate' (innerLayer)
deforestAna (Fx (inputLayer))
        = inputLayer

-- cataforward neuralnet sample desiredoutput = (snd (cata alg neuralnet)) sample

neuralnet :: IO (Fix Layer)
neuralnet =  do 
    weights_a <- randMat2D 3 3 
    weights_b <- randMat2D 3 3 
    weights_c <- randMat2D 1 3 
    let biases = replicate 3 0
        fc_network = (Fx ( Layer weights_c biases sigmoid sigmoid'
                        (Fx ( Layer  weights_b biases sigmoid sigmoid'
                            (Fx ( Layer  weights_a biases sigmoid sigmoid'
                                (Fx   InputLayer ) ) ) ) )))
    return fc_network

runFCNetwork :: [[Double]] -> [[Double]] -> IO (Fix Layer)
runFCNetwork samples desiredoutputs = do 
    network <- neuralnet 
    let trained_network = trains network samples desiredoutputs 
    return trained_network
