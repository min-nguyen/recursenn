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

module Recurrent where

import Utils
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Data.List.Split
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace

type State      = [Double]
type X          = [Double]
type Label      = [Double]
type Inputs     = [(X, Label)]


type HyperParameters = (Weights, Weights, Biases)

data ForwardProp = ForwardProp {
                        fF          :: [Double],
                        iF          :: [Double],
                        aF          :: [Double],
                        oF          :: [Double],
                        x           :: [Double],
                        h           :: [Double],
                        label       :: Label,
                        output      :: [Double],
                        stateF      :: [Double],
                        parameters  :: HyperParameters,
                        inputStack  :: Inputs
                    } 

instance Show ForwardProp where
    show (ForwardProp fF iF aF oF x h label output state parameters inputStack) 
        = "ForwardProp" ++ " f: " ++ show fF ++ " i: " ++  show iF ++  " a: " ++ 
            show aF ++ " o: " ++ show oF ++  " x: " 
            ++ show x ++  " h: " ++ show h ++  " label: " ++ show label 
            ++  " output: " ++ show output ++ " state: " ++
            show state  ++ "\n\n"

data BackProp   = BackProp {
                        deltaState_next  :: [Double],
                        deltaOut_next    :: [Double], 
                        deltaGates_next  :: [Double],  
                        forget_next      :: [Double],
                        nextLayerDeltas  :: [[Double]],
                        nextLayerWeights :: Weights
                    } deriving Show


data Weights    = Weights {
                        fW  :: [[Double]],
                        iW  :: [[Double]],
                        aW  :: [[Double]],
                        oW  :: [[Double]]
                    } | NoWeights deriving Show


data Biases     = Biases{
                        fB    :: [Double],
                        iB    :: [Double],
                        aB    :: [Double],
                        oB    :: [Double]
                    }   deriving Show

data Layer k =  Layer {
                        weights_W   :: Weights,
                        weights_U   :: Weights,
                        bias        :: Biases,
                        cells       :: Fix Cell,
                        innerLayer  :: k
                    }
                | InputLayer deriving (Functor, Show)
                
data Deltas  = Deltas {
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double]
                }
                | NoDeltas deriving Show
data Cell  k =  Cell {   
                        cellState   :: State,
                        cellDeltas  :: Deltas,
                        innerCell   :: k
                    }
                | InputCell  deriving (Functor, Show)

runs :: Layer k -> Deltas --Fix Cell --Layer k 
runs (Layer weights_W weights_U bias cells innerLayer) 
    = let initialForwardProp = ForwardProp [] [] [] [] [] [] [] [0] [0] (weights_W, weights_U, bias) [([1,2],[0.5]),([0.5,3], [1.25])]
          initialBackProp    = BackProp [0] [0] [0,0,0,0] [0] [[]] NoWeights
          (cells', forwardPropFunc) = cata alg_cell cells
          forwardProp               = forwardPropFunc [initialForwardProp]
          cells''                   = ana coalg_cell (cells', forwardProp, initialBackProp)
          (cells3, delta_total)     = cata alg2_cell cells''
         -- layer                     = updateParameters (Layer weights_W weights_U bias cells3 innerLayer) delta_total
      in  delta_total --delta_total --trace (show delta_total) layer

-- run :: Layer k -> Layer k
-- run (Layer weights_W weights_U bias cells innerLayer) 
--     = let initialForwardProp = ForwardProp [] [] [] [] [] [] [] [] [0] (weights_W, weights_U, bias) [([1,2],[0.5]),([0.5,3], [1.25])]
--           initialBackProp    = BackProp [0] [0] [[]] [[]] [] [] [[]] NoWeights
--           (cells', forwardPropFunc) = cata alg_cell cells
--           forwardProp               = forwardPropFunc [initialForwardProp]
--           cells''                   = ana coalg_cell (cells', forwardProp, id)
--           (_, _, backPropFunc)      = cata alg2_cell cells''
--           backProp                  = backPropFunc initialBackProp
--       in updateParameters (Layer weights_W weights_U bias cells'' innerLayer) backProp

alg_cell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
alg_cell (Cell state deltas (innerCell, forwardProps)) 
    = let forwardProps' = (\fps -> 
                let ForwardProp {parameters = parameters, 
                                 output = h, 
                                 stateF = prevState,
                                 inputStack = inputStack} = head fps
                    (weights_w, weights_u, biases) = parameters
                    (x, label) = head inputStack
                    f = map sigmoid $ eleadd3  (mvmul (fW weights_w) x) (mvmul (fW weights_u) h) (fB biases)
                    i = map sigmoid $ eleadd3  (mvmul (iW weights_w) x) (mvmul (iW weights_u) h) (iB biases)
                    o = map sigmoid $ eleadd3  (mvmul (oW weights_w) x) (mvmul (oW weights_u) h) (oB biases)
                    a = map tanh    $ eleadd3  (mvmul (aW weights_w) x) (mvmul (aW weights_u) h) (aB biases)
                    state'  = eleadd (elemul a i) (elemul f prevState)
                    output' = elemul o (map tanh state')
                in  ((ForwardProp f i a o x h label output' state' parameters (tail inputStack)):fps)) . forwardProps
      in  (Fx (Cell state deltas innerCell), forwardProps')
alg_cell InputCell = 
    (Fx InputCell, id)


coalg_cell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp)  
coalg_cell (Fx (Cell state deltas innerCell), forwardProps, backProp)
  = let ForwardProp f i o a x h label output updatedState hyperparameters _ = head forwardProps
        prevState =  stateF (head (tail (forwardProps))) -- 
        (weights_w, weights_u, biases) = hyperparameters
      
        BackProp dState_next deltaOut_next deltaGates_next
                    f_next 
                    nextLayerDeltas nextLayerWeightsW = backProp

        Weights fW  iW aW  oW   = weights_w
        Weights fU  iU aU  oU   = weights_u
        weightsW   = fW  ++ iW  ++ aW  ++ oW
        weightsU   = fU  ++ iU  ++ aU  ++ oU
--                         nextLayerDelta = head nextLayerDeltas
--                         deltaError = elemul (mvmul (transpose nextLayerWeightsW) nextLayerDelta) x
        deltaError = elesub output label
        dOut       = eleadd deltaError deltaOut_next
        dState     = eleadd (elemul3 dOut o (map (sub1 . sqr . tanh) updatedState)) (elemul dState_next f_next)
        
        d_a        = elemul3 dState i (map (sub1 . sqr) a)
        d_i        = elemul4 dState a i (map sub1 i)
        d_f        = elemul4 dState prevState f (map sub1 f)
        d_o        = elemul4 dOut (map tanh updatedState) o (map sub1 o)
        
        deltaGates = d_f ++ d_i ++ d_a ++ d_o
        deltaX     = mvmul (transpose weightsW) deltaGates  -- not used at the moment
        deltaOut   = mvmul (transpose weightsU) deltaGates

        deltaW     = outerProduct deltaGates  x 
        deltaU     = outerProduct deltaGates_next output
        deltaB     = deltaGates
        backProp'  = BackProp dState deltaOut deltaGates f nextLayerDeltas nextLayerWeightsW

      in Cell updatedState (Deltas deltaW deltaU deltaB) 
                        (innerCell, tail forwardProps, backProp' )
coalg_cell (Fx InputCell, forwardProps, backProp)
    = InputCell


alg2_cell ::  Cell (Fix Cell, Deltas) ->  (Fix Cell, Deltas)
alg2_cell (Cell state deltas (innerCell, delta_total)) 
    =   
        let Deltas {deltaW = deltaW1, deltaU = deltaU1, deltaB = deltaB1} = deltas
            Deltas {deltaW = deltaW2, deltaU = deltaU2, deltaB = deltaB2} = 
                    case delta_total of NoDeltas -> Deltas [[0,0],[0,0],[0,0],[0,0]] 
                                                           [[0],[0],[0],[0]] [0,0,0,0]
                                        _        -> delta_total
                
            deltaW_total = (eleaddM deltaW1 deltaW2)
            deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
            deltaB_total = (eleadd deltaB1 deltaB2)
            delta_total' = Deltas deltaW_total deltaU_total deltaB_total
        in  trace (show (deltaW1, deltaU1, deltaB1) ++ "\n") (Fx (Cell state deltas innerCell ), delta_total') --
alg2_cell InputCell 
    = (Fx InputCell, NoDeltas)



updateParameters ::  Layer k -> Deltas -> Layer k
updateParameters (Layer weights_w weights_u biases cells innerLayer) delta_total
    =   let Deltas deltaW_total deltaU_total deltaB_total = delta_total

            Weights fW iW aW oW = weights_w
            Weights fU iU aU oU = weights_u
            Biases  fB iB aB oB = biases
            weightw_length      = length fW
            weightu_length      = length fU
            biases_length       = length fB
            p = elesub3 [fW  , iW  , aW  , oW] (map  ((map2 (0.1 *)) . (chunksOf weightw_length)) deltaW_total)
            q = elesub3 [fU, iU, aU, oU] (map  ((map2 (0.1 *)) . (chunksOf weightu_length)) deltaU_total)
            r = elesubm [fB, iB, aB, oB] (((map2 (0.1 *)) . (chunksOf biases_length)) deltaB_total)
            --weights_w'  = Weights fW' iW' aW' oW'
            weights_w'  = Weights [[]] [[]] [[]] [[]]
            weights_u'  =  Weights [[]] [[]] [[]] [[]] --Weights fU' iU' aU' oU'
            biases'     = Biases [] [] [] [] --fB' iB' aB' oB'
        in trace (show (delta_total)) (Layer weights_w' weights_u' biases' cells innerLayer)
-- [fB', iB', aB', oB'] 
--[fU', iU', aU', oU']
--[fW', iW', aW', oW']
example = Layer (Weights [[0.7, 0.45]]  [[0.95, 0.8]]  [[0.45, 0.25]]   [[0.6, 0.4]])
                (Weights [[0.1]]        [[0.8]]         [[0.15]]         [[0.25]])
                (Biases   [0.15]        [0.65]          [0.2]            [0.1])
                (Fx (Cell [0.68381] NoDeltas (Fx (Cell [0] NoDeltas (Fx InputCell))))) (Fx InputLayer)