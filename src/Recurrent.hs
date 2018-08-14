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
import Prelude
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

data Cell  k =   Cell {   
                        cellState   :: State,
                        cellDeltas  :: Deltas,
                        innerCell   :: k
                     }
                | EndCell {
                        cellState   :: State,
                        cellDeltas  :: Deltas,
                        innerCell   :: k
                     }
                | InputCell  deriving (Functor, Show)

runs :: Layer k ->  Layer k 
runs (Layer weights_W weights_U bias cells innerLayer) 
    = let initialForwardProp = ForwardProp [] [] [] [] [] [] [] [0] [0] (weights_W, weights_U, bias) [([1,2],[0.5]),([0.5,3], [1.25])]
          initialBackProp    = BackProp [0] [0] [0,0,0,0] [0] [[]] NoWeights

          (cells1, forwardPropFunc) = cata alg_cell cells
          forwardProp               = forwardPropFunc [initialForwardProp]

          cells2                    = ana coalg_cell (cells1, forwardProp, initialBackProp)
          (cells3, deltaTotalFunc)  = cata alg2_cell cells2

          deltaTotal                = deltaTotalFunc $ Deltas (fillMatrix (4 * (length $ fW weights_W)) (length $ fst $ head $ inputStack initialForwardProp) 0.0) 
                                                              (fillMatrix (4 * (length $ fW weights_W)) (length $ fW weights_W) 0.0)
                                                              (replicate  (4 * (length $ fW weights_W)) 0.0)
          layer                     = updateParameters (Layer weights_W weights_U bias cells3 innerLayer) deltaTotal
      in  layer 

alg_cell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
alg_cell InputCell = 
    (Fx InputCell, id)
alg_cell cell
    = let (state, deltas, (nextCell, forwardProps)) = (cellState cell, cellDeltas cell, innerCell cell)
          forwardProps' = (\fps -> 
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
      in  (Fx (cell {innerCell = nextCell}), forwardProps')

coalg_cell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp) 
coalg_cell (Fx (EndCell state deltas innerCell), forwardProps, backProp)
  = let ForwardProp f i o a x h label output updatedState hyperparameters _ = head forwardProps
        prevState =  stateF (head (tail (forwardProps))) 
        (weights_w, weights_u, biases) = hyperparameters
        Weights fW  iW aW  oW   = weights_w
        Weights fU  iU aU  oU   = weights_u
        weightsW   = fW  ++ iW  ++ aW  ++ oW
        weightsU   = fU  ++ iU  ++ aU  ++ oU
--                         nextLayerDelta = head nextLayerDeltas
--                         deltaError = elemul (mvmul (transpose nextLayerWeightsW) nextLayerDelta) x
        deltaError = elesub output label
        dOut       = deltaError
        dState     = (elemul3 dOut o (map (sub1 . sqr . tanh) updatedState))
        
        d_a        = elemul3 dState i (map (sub1 . sqr) a)
        d_i        = elemul4 dState a i (map sub1 i)
        d_f        = elemul4 dState prevState f (map sub1 f)
        d_o        = elemul4 dOut (map tanh updatedState) o (map sub1 o)
        
        deltaGates = d_f ++ d_i ++ d_a ++ d_o
        deltaX     = mvmul (transpose weightsW) deltaGates  -- not used at the moment
        deltaOut   = mvmul (transpose weightsU) deltaGates

        deltaW     = outerProduct deltaGates  x 
        deltaU     = fillMatrix (4 * (length fU)) (length $ head fU) (0.0 :: Double)
        deltaB     = deltaGates
        backProp'  = BackProp dState deltaOut deltaGates f (nextLayerDeltas backProp) (nextLayerWeights backProp)

    in EndCell updatedState (Deltas deltaW deltaU deltaB) 
                        (innerCell, tail forwardProps, backProp' )

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

      in trace (show deltaU) $ Cell updatedState (Deltas deltaW deltaU deltaB) 
                        (innerCell, tail forwardProps, backProp' )
coalg_cell (Fx InputCell, forwardProps, backProp)
    = InputCell


alg2_cell ::  Cell (Fix Cell, Deltas -> Deltas) ->  (Fix Cell, Deltas -> Deltas)
alg2_cell InputCell 
    = (Fx InputCell, id)

alg2_cell cell
    =   let (state, deltas, (nextCell, deltaTotalFunc)) = (cellState cell, cellDeltas cell, innerCell cell)
            Deltas {deltaW = deltaW1, deltaU = deltaU1, deltaB = deltaB1} = deltas
            deltaTotalFunc' = (\deltaTotal -> 
                let Deltas {deltaW = deltaW2, deltaU = deltaU2, deltaB = deltaB2} = deltaTotal

                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
                    deltaB_total = (eleadd deltaB1 deltaB2)
                in  Deltas deltaW_total deltaU_total deltaB_total) . deltaTotalFunc
        in  (Fx (cell {innerCell = nextCell}), deltaTotalFunc') --



updateParameters ::  Layer k -> Deltas -> Layer k
updateParameters (Layer weights_w weights_u biases cells innerLayer) delta_total
    =   let Deltas deltaW_total deltaU_total deltaB_total = delta_total

            Weights fW iW aW oW = weights_w
            Weights fU iU aU oU = weights_u
            Biases  fB iB aB oB = biases
            weightw_length      = length fW
            weightu_length      = length fU
            biases_length       = length fB
            (wf:wi:wa:wo:_)     = map cons $ elesubm (fW ++ iW ++ aW ++ oW) (map2 (0.1 *) deltaW_total)
            (uf:ui:ua:uo:_)     = map cons $ elesubm (fU ++ iU ++ aU ++ oU) (map2 (0.1 *) deltaU_total)
            (bf:bi:ba:bo:_)     = map cons $ elesub  (fB ++ iB ++ aB ++ oB) (map (0.1 *) deltaB_total)
            
            weights_w'  =  Weights wf wi wa wo
            weights_u'  =  Weights uf ui ua uo 
            biases'     =  Biases  bf bi ba bo 
        in trace (show  deltaU_total) (Layer weights_w' weights_u' biases' cells innerLayer)

example = Layer (Weights [[0.7, 0.45]]  [[0.95, 0.8]]  [[0.45, 0.25]]   [[0.6, 0.4]])
                (Weights [[0.1]]        [[0.8]]         [[0.15]]         [[0.25]])
                (Biases   [0.15]        [0.65]          [0.2]            [0.1])
                (Fx (EndCell [0.68381] NoDeltas (Fx (Cell [0] NoDeltas (Fx InputCell))))) (Fx InputLayer)