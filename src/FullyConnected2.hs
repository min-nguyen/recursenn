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

module FullyConnected2 where

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
                                      in  trace ((\z -> showFullPrecision $ abs $ read $ formatFloatN (z/100) 8) $ head (zipWith (-) outputs desiredOutput)) cost
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

-- train :: Fix Layer -> Inputs -> DesiredOutput -> Fix Layer 
-- train neuralnet sample desiredoutput 
--     =  meta alg (\(nn, diff_fun) -> (nn, BackPropData (diff_fun [sample]) desiredoutput [] [[]] )) coalg $ neuralnet
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

construct :: [(Weights, Biases, Activation, Activation')] -> Fix Layer
construct (x:xs) = Fx (Layer weights biases activation activation' (construct (xs)))
            where (weights, biases, activation, activation') = x
construct []       = Fx InputLayer

-- cataforward neuralnet sample desiredoutput = (snd (cata alg neuralnet)) sample


example =  (Fx ( Layer [[0.4, 0.1, 0.3]] [0, 0, 0] sigmoid sigmoid'
            (Fx ( Layer  [[0.1,0.2,0.5],[0.4,0.2,0.4],[0.3,0.8,0.12]] [0, 0, 0] sigmoid sigmoid'
            (Fx ( Layer  [[0.3,0.6,0.15],[0.2,0.1,0.7],[0.6,0.5,0.2]] [0, 0, 0] sigmoid sigmoid'
             (Fx   InputLayer ) ) ) ) )))

example' =  (Fx ( Layer [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] [0, 0, 0] sigmoid sigmoid'
            (Fx ( Layer  [[3.0,6.0,2.0],[2.0,1.0,7.0],[6.0,5.0,2.0]] [0, 0, 0] sigmoid sigmoid'
             (Fx   InputLayer ) ) ) ) )


runFullyConnected = print $ show $ let nn = (train example [-0.5, 0.2, 0.5] [0.0, 1.0, 0.0]) 
                                   in nn -- train nn loss [1.0, 2.0, 0.2] [-26.0, 5.0, 3.0]

-- runFullyConnectedForward = cataforward example [[-0.5, 0.2, 0.5]] [0.0, 1.0, 0.0]

runSineNetwork samples desiredoutputs = trains example samples desiredoutputs -- [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]] [[0.85],[0.991]]

exampleSineNetwork = Fx ( Layer w2 b2 sigmoid sigmoid' 
                        (Fx (Layer w1 b1 sigmoid sigmoid' 
                            (Fx InputLayer))))
                    where w1 = [[0.12,0.87,0.28,0.39,0.40,0.88,0.07,0.85,0.12,0.10],
                                [0.81,0.59,0.71,0.87,0.68,0.73,0.39,0.29,0.07,0.58],
                                [0.89, 0.45,0.91,0.60,0.61,0.81,0.34,0.09,0.78,0.75],
                                [0.12,0.53,0.72,0.15,0.21,0.34,0.79,0.98,0.92,0.97],
                                [0.54,0.92,0.50,0.48,0.60,0.77,0.43,0.76,0.37,0.19],
                                [0.76,0.21,0.84,0.17,0.39,0.38,0.84,0.72,0.24,0.34],
                                [0.98,0.09,0.32,0.11,0.80,0.78,0.90,0.53,0.42,0.32],
                                [0.82,0.76,0.50,0.10,0.29,0.20,0.96,0.23,0.92,0.32],
                                [0.97,0.52,0.00,0.70,0.68,0.33,0.42,0.47,0.27,0.37],
                                [0.61,0.79,0.96,0.19,0.74,0.30,0.36,0.63,0.44,0.80]]
                          w2 =  [[0.19,0.58,0.64,0.18,0.50,0.08,0.40,0.65,0.52,0.21]]  
                          b1  = [0,0,0,0,0,0,0,0,0,0]
                          b2  = [0]                            
-- 0.668187772
-- 0.937026644
-- 0.2689414214

-- 0.975377100
-- 0.895021978
-- 0.956074004

-- a^L - y
-- 0.9753771
-- -0.104978022
-- 0.956074004

-- z 
-- 0.69999999924
-- 2.70000000097
-- -0.99999999984

-- sigmoid'(z)
-- 0.24431158873
-- 0.18050777912
-- 0.23857267148

-- delta
-- 0.23829592891186
-- -0.0189493496076305
-- 0.0641620733750264

forward' :: Inputs -> Weights -> Biases -> (Double -> Double) -> Inputs
forward' inputs weights biases activate = map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (inputs))) weights) biases))

runOneLayer = forward' [-0.5, 0.2, 0.5] [[3.0,6.0,2.0],[2.0,1.0,7.0],[6.0,5.0,2.0]] [0,0,0] sigmoid 

runTwoLayer = forward' [0.668187772, 0.937026644, 0.2689414214] [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] [0,0,0] sigmoid 