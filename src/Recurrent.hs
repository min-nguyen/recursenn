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
                        state       :: [Double],
                        parameters  :: HyperParameters,
                        inputStack  :: Inputs
                    } 

instance Show ForwardProp where
    show (ForwardProp fF iF aF oF x h label output state parameters inputStack) 
        = "ForwardProp" ++ " f: " ++ show fF ++ " i: " ++  show iF ++  " a: " ++ 
            show aF ++ " o: " ++ show oF ++  " x: " 
            ++ show x ++  " h: " ++ show h ++  " label: " ++ show label 
            ++  " output: " ++ show output ++ " state: " ++
            show state ++ " parameters: " ++ show parameters ++ show inputStack ++ "\n\n"


data BackProp   = BackProp {
                        deltaState_next  :: [Double],
                        deltaOut_next    :: [Double],   
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double],
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
                
data Cell  k =  Cell {   
                        cellState   :: State,
                        innerCell   :: k
                    }
                | InputCell  deriving (Functor)

instance Show k => Show (Cell k) where
    show (Cell cellState innercell) = show cellState ++ "\n" ++ show innercell
    show (InputCell) = "InputCell"

runs :: Layer k -> [ForwardProp]
runs (Layer weights_W weights_U bias cells innerLayer) 
    = let initialForwardProp = ForwardProp [] [] [] [] [] [] [] [0] [0] (weights_W, weights_U, bias) [([1,2],[0.5]),([0.5,3], [1.25])]
          initialBackProp    = BackProp [0] [0] [[]] [[]] [] [] [[]] NoWeights
          (cells', forwardPropFunc) = cata alg_cell cells
          forwardProp               = forwardPropFunc [initialForwardProp]
      in forwardProp

run :: Layer k -> Layer k
run (Layer weights_W weights_U bias cells innerLayer) 
    = let initialForwardProp = ForwardProp [] [] [] [] [] [] [] [] [0] (weights_W, weights_U, bias) [([1,2],[0.5]),([0.5,3], [1.25])]
          initialBackProp    = BackProp [0] [0] [[]] [[]] [] [] [[]] NoWeights
          (cells', forwardPropFunc) = cata alg_cell cells
          forwardProp               = forwardPropFunc [initialForwardProp]
          cells''                   = ana coalg_cell (cells', forwardProp, id)
          (_, _, backPropFunc)      = cata alg2_cell cells''
          backProp                  = backPropFunc initialBackProp
      in updateParameters (Layer weights_W weights_U bias cells'' innerLayer) backProp

alg_cell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
alg_cell (Cell state (innerCell, forwardProps)) 
    = let forwardProps' = (\fps -> 
                let ForwardProp {parameters = parameters, 
                                 output = h, 
                                 state = prevState,
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
      in  (Fx (Cell state innerCell), forwardProps')
alg_cell InputCell = 
    (Fx InputCell, id)




coalg_cell :: (Fix Cell, [ForwardProp], BackProp -> BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp -> BackProp)  
coalg_cell (Fx (Cell state innerCell), forwardProps, _)
    = let ForwardProp f i o a x h label output updatedState hyperparameters _ = head forwardProps
          ForwardProp {state = prevState}                                     = head (tail (forwardProps ++ []))
          (weights_w, weights_u, biases) = hyperparameters
          backProp' = (\bp -> 
                    let BackProp dState_next deltaOut_next 
                                 deltaW_next deltaU_next deltaB_next f_next 
                                 nextLayerDeltas nextLayerWeightsW = bp

                        Weights aW  iW fW  oW   = weights_w
                        Weights aU  iU fU  oU   = weights_u
                        weightsW   = aW  ++ iW  ++ fW  ++ oW
                        weightsU   = aU  ++ iU  ++ fU  ++ oU

                        deltaError = elesub output label
                        dOut       = eleadd deltaError deltaOut_next
                        dState     = eleadd (elemul3 dOut o (map (sub1 . sqr . tanh) state)) (elemul dState_next f_next)
                        
                        d_a        = elemul3 dState i (map (sub1 . sqr) a)
                        d_i        = elemul4 dState a i (map sub1 i)
                        d_f        = elemul4 dState prevState f (map sub1 f)
                        d_o        = elemul4 dOut (map tanh state) o (map sub1 o)
                        
                        deltaGates = d_a ++ d_i ++ d_f ++ d_o
                        deltaX     = mvmul (transpose weightsW) deltaGates  -- not used at the moment
                        deltaOut   = mvmul (transpose weightsU) deltaGates

                        deltaW     = mmmul (map cons deltaGates) (cons x) 
                        deltaU     = mmmul (map cons deltaGates) (cons output) 
                        deltaB     = deltaGates

                    in  BackProp dState deltaOut deltaW deltaU deltaB f (tail (nextLayerDeltas ++ [])) nextLayerWeightsW)
      in  Cell updatedState (innerCell, tail forwardProps, backProp' )
coalg (Fx InputCell, forwardProps, backProp)
    = InputCell
-- coalg_cell (Fx (Cell state innerCell), forwardProps, _)
--     = let ForwardProp f i o a x h label output updatedState hyperparameters _ = head forwardProps
--           ForwardProp {state = prevState}                                     = head (tail forwardProps)
--           (weights_w, weights_u, biases) = hyperparameters
--           backProp' = (\bp -> 
--                     let BackProp dState_next deltaOut_next 
--                                  deltaW_next deltaU_next deltaB_next f_next
--                                  nextLayerDeltas nextLayerWeightsW = bp

--                         Weights fW iW aW oW   = weights_w
--                         Weights fU iU aU oU   = weights_u
--                         weightsW   = aW  ++ iW  ++ fW  ++ oW
--                         weightsU   = aU  ++ iU  ++ fU  ++ oU

--                         nextLayerDelta = head nextLayerDeltas
--                         deltaError = elemul (mvmul (transpose nextLayerWeightsW) nextLayerDelta) x

--                         dOut       = eleadd deltaError deltaOut_next
--                         dState     = eleadd (elemul3 dOut o (map (sub1 . sqr . tanh) state)) (elemul dState_next f_next)
                        
--                         d_a        = elemul3 dState i (map (sub1 . sqr) a)
--                         d_i        = elemul4 dState a i (map sub1 i)
--                         d_f        = elemul4 dState prevState f (map sub1 f)
--                         d_o        = elemul4 dOut (map tanh state) o (map sub1 o)
--                         deltaGates = d_a ++ d_i ++ d_f ++ d_o
                        
--                         deltaX     = mvmul (transpose weightsW) deltaGates  -- not used at the moment
--                         deltaOut   = mvmul (transpose weightsU) deltaGates

--                         deltaW     = mmmul (map cons deltaGates) (cons x) 
--                         deltaU     = mmmul (map cons deltaGates) (cons output) 
--                         deltaB     = deltaGates

--                     in  BackProp dState deltaOut f deltaW deltaU deltaB (tail nextLayerDeltas) nextLayerWeightsW)
--       in  Cell updatedState (innerCell, tail forwardProps, backProp' )


alg2_cell ::  Cell (Fix Cell, [ForwardProp],  BackProp -> BackProp) -> (Fix Cell, [ForwardProp],  BackProp -> BackProp)
alg2_cell (Cell state (innerCell, forwardProps, backProp)) 
    =   let backProp' = \bp -> 
                let BackProp {deltaW = deltaW1, deltaU = deltaU1, deltaB = deltaB1} = bp
                    BackProp {deltaW = deltaW2, deltaU = deltaU2, deltaB = deltaB2} = backProp bp
                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2)
                    deltaB_total = (eleadd  deltaB1 deltaB2)
                in  bp {deltaW = deltaW_total, deltaU = deltaU_total, deltaB = deltaB_total}
        in (Fx (Cell state innerCell), forwardProps, backProp') 
alg2 InputCell 
    = (Fx InputCell, [], id)



updateParameters ::  Layer k -> BackProp -> Layer k
updateParameters (Layer weights_w weights_u biases cells innerLayer) backProp
    =   let (deltaW_total, deltaU_total, deltaB_total) = (deltaW backProp, deltaU backProp, deltaB backProp)
            [dfW, diW, daW, doW] = deltaW_total
            [dfU, diU, daU, doU] = deltaU_total
            [dfB, diB, daB, doB] = deltaB_total
            Weights fW iW aW oW = weights_w
            Weights fU iU aU oU = weights_u
            Biases  fB iB aB oB = biases
            weightw_length      = length fW
            weightu_length      = length fU
            biases_length       = length fB
            [fW', iW', aW', oW'] = elesub3 [fW, iW, aW, oW] (map  ((map2 (0.1 *)) . (chunksOf weightw_length)) [dfW, diW, daW, doW])
            [fU', iU', aU', oU'] = elesub3 [fU, iU, aU, oU] (map  ((map2 (0.1 *)) . (chunksOf weightu_length)) [dfU, diU, daU, doU])
            [fB', iB', aB', oB'] = elesubm [fB, iB, aB, oB] (((map2 (0.1 *)) . (chunksOf biases_length)) deltaB_total)
            weights_w'  = Weights fW' iW' aW' oW'
            weights_u'  = Weights fU' iU' aU' oU'
            biases'     = Biases  fB' iB' aB' oB'
        in  (Layer weights_w' weights_u' biases' cells innerLayer)


example = Layer (Weights [[0.7, 0.45]]  [[0.95, 0.8]]  [[0.45, 0.25]]   [[0.6, 0.4]])
                (Weights [[0.1]]        [[0.8]]         [[0.15]]         [[0.25]])
                (Biases   [0.15]        [0.65]          [0.2]            [0.1])
                (Fx (Cell [0.68381] (Fx (Cell [0] (Fx InputCell))))) (Fx InputLayer)