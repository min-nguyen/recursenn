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

module Convolution where

import Lib
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List.Split
import Data.Ord
import Text.Show.Functions
import qualified Vector as Vector
import Vector (Vector((:-)))
import Debug.Trace

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                          Alg & Coalg                           | | ---
  --- ‾------------------------------------------------------------------‾---

flatten :: Fractional a => [[a]] -> Int -> [[a]]
flatten image n =
    let splitVertical image' stackArray =   
                                if length image' < n 
                                then stackArray
                                else (splitHorizontal image' (take n image') stackArray)
        splitHorizontal image'' imageChunk stack' = case () of 
                                _ | length (head imageChunk) < n -> (splitVertical (tail image'') stack')
                                _ | otherwise -> let new_stack = (stack' ++ (concat $ map (take n) $ take n imageChunk))
                                                 in  splitHorizontal image'' (map tail imageChunk) new_stack
    in chunksOf (n*n) (splitVertical image [])

convolute2D :: Fractional a => [[a]] -> [[a]] -> [[a]]
convolute2D filter image 
    = let flat_image = flatten image (length filter)
      in  chunksOf (length filter) $ map  (sum . (zipWith (*) (concat filter))) flat_image

convolute3D :: Fractional a => [[[a]]] -> [[[a]]] -> [[a]]
convolute3D filter image 
    = let n     = length filter
          bias  = 1.0
      in  map (map (bias + )) $ foldr eleaddm (fillMatrix n n 0.0) [  convolute2D filter2d image2d  |  (image2d, filter2d) <- (zip image filter)]

-- Each filter is applied to the entire depth of images in the current stack of 2D images, so
-- must have the same depth as the input volume
-- Each filter will produce a separate 2D activation map (output volume)
-- Each output volume by a filter, can be interpreted as an output of a neuron
-- Given a receptive field size of 3x3 and input volume of 16x16x20, every neuron in the conv layer
-- would now have a total of 3*3*20 = 180 connections to the input volume. The connectivity is local
-- in space (3x3) but along the full input depth (20).
-- The depth of the output volume is the number of filters we used

-- alg :: CNNLayer (Fix CNNLayer, ([ImageStack] -> [ImageStack]) ) -> CNNLayer (Fix CNNLayer, ([ImageStack] -> [ImageStack]))
-- alg ConvolutionalLayer filters biases (innerLayer, forwardPass)
--        = (\imageStacks -> 
--             let inputVolume = (head imageStacks) 
--                 [map (convolute filter) inputVolume | filter <- filters]) . forwardPass




-- coalgx :: (Fix Layer, BackPropData) -> (BackPropData -> Layer  (Fix Layer, BackPropData) )
-- coalgx (Fx (Layer weights biases (activate, activate') innerLayer), (BackPropData { inputStack = (output:input:xs), .. }))
--     =  \backPropData -> let (newWeights, newBiases) = (backward weights biases input backPropData)
--                         in Layer newWeights newBiases (activate, activate') (innerLayer, backPropData {outerWeights = weights})
-- coalgx (Fx InputLayer, output)
--     =  \_ -> InputLayer 

-- coalgy :: (Fix Layer, BackPropData) -> BackPropData
-- coalgy (Fx (Layer weights biases (activate, activate') innerLayer), backPropData)
--     =   let BackPropData { inputStack = (outputs:inputs:xs), .. } = backPropData
--             delta = compDelta activate' inputs outputs backPropData
--         in  backPropData { inputStack = (inputs:xs), outerDeltas = delta }
-- coalgy (Fx InputLayer, backPropData)
--     =   backPropData

-- ---- |‾| -------------------------------------------------------------- |‾| ----
--  --- | |                    Forward & Back Propagation                  | | ---
--   --- ‾------------------------------------------------------------------‾---

-- compDelta ::  Activation' -> Inputs -> Outputs -> BackPropData -> Deltas 
-- compDelta derivActivation inputs outputs (BackPropData _ finalOutput desiredOutput outerDeltas outerWeights)   
--     = case outerDeltas of  [] -> elemul (map (\x -> x*(x-1)) outputs) (zipWith (-) outputs desiredOutput)
--                            _  -> elemul (mvmul (transpose outerWeights) outerDeltas) (map derivActivation inputs)

-- forward :: Weights -> Biases -> Activation -> ([Inputs] -> [Inputs]) -> ([Inputs] -> [Inputs])
-- forward weights biases activate k 
--     = (\inputs -> (map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

-- backward :: Weights -> Biases -> Inputs  -> BackPropData -> (Weights, Biases)
-- backward weights biases inputs (BackPropData {outerDeltas = updatedDeltas, ..} )
--     = let learningRate = 0.2
--           inputsDeltasWeights = map (zip3 inputs updatedDeltas) weights
--           updatedWeights = [[ w - learningRate*d*i  |  (i, w, d) <- idw_vec ] | idw_vec <- inputsDeltasWeights]                                                  
--           updatedBiases  = map (learningRate *) updatedDeltas
--       in (updatedWeights, updatedBiases)

-- ---- |‾| -------------------------------------------------------------- |‾| ----
--  --- | |                    Running And Constructing NNs                | | ---
--   --- ‾------------------------------------------------------------------‾---

-- -- trainc :: Fix Layer -> LossFunction -> Inputs -> DesiredOutput -> Fix Layer 
-- -- trainc neuralnet lossfunction sample desiredoutput 
-- --     = trace (show $ head inputStack) $ 
-- --         ana coalg $ (nn, )
-- --             where 
-- --                 (nn, diff_fun)      = cata alg neuralnet
-- --                 inputStack   = diff_fun [sample]

-- example =  (Fx ( Layer [[3.0,6.0,2.0],[2.0,1.0,7.0],[6.0,5.0,2.0]] [0, 0, 0] (sigmoid, sigmoid')
--             (Fx ( Layer [[4.0,0.5,2.0],[1.0,1.0,2.0],[3.0,0.0,4.0]] [0, 0, 0] (sigmoid, sigmoid')
--              (Fx   InputLayer ) ) ) ) )

-- main = print $ show $ trainc example loss [1.0, 2.0, 0.2] [-26.0, 5.0, 3.0]


