
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

module Recurrent where
import Prelude
import Utils
import Control.Applicative
import Data.Functor     
import Data.Foldable
import Data.Maybe
import qualified Data.Functor.Fixedpoint    as F
import Data.Traversable
import Control.Lens hiding (snoc,cons)
import Data.List
import qualified Data.Vec as V
import Data.Ord
import Data.List.Split
import Text.Show.Functions
import Debug.Trace

type State      = [Double]
type X          = [Double]
type Label      = [Double]
type Inputs     = [(X, Label)]

--f i a o

type Gates      =  V.Vec4 [Double] 
type Weights    =  V.Vec4 [[Double]] 
type Biases     =  V.Vec4 [Double] 

type HyperParameters = (Weights, Weights, Biases)

data ForwardProp = ForwardProp {
                        _gates       :: Gates,
                        _input       :: [Double],
                        _des_out     :: Label,
                        _output      :: [Double],
                        _state       :: [Double], -- change this to just state
                        _params      :: HyperParameters,
                        _inputStack  :: Inputs
                    } deriving Show
makeLenses ''ForwardProp

data BackProp   = BackProp {
                        _nextDState         :: [Double],
                        _nextDOut           :: [Double], 
                        _nextDGates         :: [Double],  
                        _nextF              :: [Double],
                        _nextLayerDXs       :: Maybe [[Double]]
                    } deriving Show
makeLenses ''BackProp




data Deltas  = Deltas {
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double],
                        deltaXs          :: [[Double]]
                }
                | NoDeltas

instance Show Deltas where 
    show (Deltas w u b x) =
        "Deltas: \n" ++ "DeltaW: " ++ show w ++ "\n" ++ "DeltaU: " ++ show u ++ "\n" ++
        "DeltaB: " ++ show b ++ "\n" ++ "DeltaX: " ++ show x ++ "\n"
    show NoDeltas = "NoDeltas \n"

data Cell  k =   Cell {   
                        _cellState   :: State,
                        _cellDeltas  :: Deltas,
                        _innerCell   :: k
                     }
                | EndCell {
                        _cellState   :: State,
                        _cellDeltas  :: Deltas,
                        _innerCell   :: k
                     }
                | InputCell  deriving (Functor)
makeLenses ''Cell

data Layer k =  Layer {
                        _hparams     :: HyperParameters,
                        _cells       :: Fix Cell,
                        _innerLayer  :: k
                    }
                | InputLayer deriving (Functor, Foldable, Traversable)
makeLenses ''Layer

instance Show k => Show (Cell k) where 
    show (Cell cstate cdeltas  inner_cell ) =
        "Cell: \n" ++ "State: " ++ show cstate ++ "\n" ++ "Deltas: " ++ show cdeltas ++ "\n"
        ++ show inner_cell
    show (EndCell cstate cdeltas  inner_cell ) =
        "EndCell: \n" ++ "State: " ++ show cstate ++ "\n" ++ "Deltas: " ++ show cdeltas ++ "\n"
        ++ show inner_cell 
    show InputCell = "InputCell \n"


instance Show k => Show (Layer k) where
    show (Layer hparam cell inner_layer) = 
        "Layer\n" ++ "Hyperparameters: " ++ show hparam ++ "\n" ++ "Cells: \n" ++ show cell ++
        "\n" ++ show inner_layer
    show InputLayer = "InputLayer \n"

trains :: Fix Layer -> [Inputs] -> Fix Layer
trains neuralnet samples   
    = foldr (\sample nn -> 
                  let updatedNetwork = runLayer nn sample 
                  in  updatedNetwork) neuralnet samples

runLayer :: Fix Layer  -> Inputs -> Fix Layer
runLayer layer sample = let f = \(layer', forwardProp') -> (layer', forwardProp' sample, (initBackProp 1 2 Nothing Nothing))
                         in  meta algLayer f coalgLayer layer

algLayer :: Layer (Fix Layer, Inputs -> [[ForwardProp]]) -> (Fix Layer, Inputs -> [[ForwardProp]])
algLayer InputLayer = (Fx InputLayer, (\sample -> cons [ForwardProp emptyGates [] l x [0] emptyParams sample | (x,l) <- sample  ]))
algLayer (Layer params cells (innerLayer, nextForwardProp))
    = let forwardProp  = (\fps ->
                    let fp = head fps
                        inputs = map tuplify2 $ chunksOf 2 $ fp <**> [(^.output), (^.des_out)] 
                        (hDim, dDim)       = let w = params^._1 in (length $ w ! 1, length $ head $ w ! 1)
                        initialForwardProp = initForwardProp hDim dDim params inputs
                        (cell, fpFunc)     = cata algCell cells 
                        layerFP            = fpFunc [initialForwardProp]
                        showinputs =  map (\l -> l ^. input) layerFP
                        showoutputs =  map (\l -> l ^. output) layerFP
                    in  --trace (show showinputs ++ "\n" ++ show showoutputs) 
                        (layerFP:fps)) . nextForwardProp
      in (Fx (Layer params cells innerLayer), forwardProp) 

coalgLayer :: (Fix Layer, [[ForwardProp]], BackProp) -> Layer (Fix Layer, [[ForwardProp]], BackProp)
coalgLayer (Fx InputLayer, fp, bp)
    = InputLayer
coalgLayer (Fx (Layer params cells innerLayer), fps, backProp)
    =   let w                   = params^._1
            (hDim, dDim)        = (length $ w ! 1, length $ head $ w ! 1)
            (cell, deltaFunc)   = hylo algCell2 coalgCell (cells, head fps, backProp)
            deltaTotal          = deltaFunc (initDelta hDim dDim)

            backProp'           = initBackProp hDim dDim (Just $ deltaXs deltaTotal) (Just $ w)
            showcost            = trace ((\z -> showFullPrecision $ read $ formatFloatN (z/100) 8) $ sum $ map abs $ concat $ deltaW deltaTotal) 

        in  case innerLayer of (Fx (InputLayer)) -> showcost $  updateParameters (Layer params cell (innerLayer, tail fps, backProp')) deltaTotal
                               _             -> showcost (updateParameters (Layer params cell (innerLayer, tail fps, backProp')) deltaTotal)

algCell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
algCell InputCell = 
    (Fx InputCell, id)
algCell cell
    = let (nextCell, forwardProps) = (_innerCell cell)
          forwardProps' = (\fps -> 
                let fp = head fps
                    (x, label) = head (fp^.inputStack)
                    gates   = compGates (fp^.params) x (fp^.output)
                    state'  = eleadd (elemul (gates ! 3) (gates ! 2)) (elemul (gates ! 1) (fp^.state))
                    output' = elemul (gates ! 4) (map tanh state')
                in  ((ForwardProp gates x label output' state'  (fp^.params) (tail (fp^.inputStack))):fps)) . forwardProps
      in  (Fx (cell & innerCell .~ nextCell), forwardProps')


-- Do i need to add deltaX and deltaOut to produce the real deltaOut at every cell??

coalgCell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp) 
coalgCell (Fx InputCell, forwardProps, backProp)
    = InputCell
coalgCell (Fx cell, forwardProps, backProp)
  = let fp = head forwardProps
        lastState               = (head (tail forwardProps)) ^. state 
        (gate, updatedState)    = (fp ^. gates, fp ^. state)
        (weightsW, weightsU)    = mapT2 (V.foldr (++) [[]]) (fp^.params._1, fp^.params._2)

        BackProp dState_next deltaOut_next deltaGates_next f_next _  = backProp

        error = (elesub (fp^.output)  (fp^.des_out))

        dOut =  
            case (cell) 
            of  (EndCell {})  -> (elesub (fp^.output)  (fp^.des_out)) 
                (Cell {})     -> eleadd deltaOut_next (elesub (fp^.output)  (fp^.des_out))

        deltaState = eleadd (elemul3 dOut (gate ! 4) (map (sub1 . sqr . tanh) updatedState)) (elemul dState_next f_next)
        deltaGates = compDGates gate dOut deltaState updatedState lastState 
        (deltaX, deltaOut)     = mapT2 (mvmulk deltaGates . transpose) (weightsW, weightsU)

        deltaW     = outerProduct deltaGates (fp^.input) 
        deltaU     = 
            case cell
            of  EndCell {} -> fillMatrix (length deltaGates) (quot (length deltaGates) 4) 0.0
                Cell {}    -> outerProduct deltaGates_next (fp^.output)
        deltaB     = deltaGates

        backProp'  = backProp   & nextDState .~ deltaState 
                                & nextDOut   .~ deltaOut 
                                & nextDGates .~ deltaGates 
                                & nextF      .~ (gate ! 1)
                                & nextLayerDXs .~   case backProp ^. nextLayerDXs 
                                                    of   Just dxs -> Just (tail dxs) 
                                                         Nothing  -> Nothing
    in  
        (cell & cellState .~ updatedState
              & cellDeltas .~ (Deltas deltaW deltaU deltaB [deltaX])
              & innerCell .~ (fromJust (cell ^? innerCell), tail forwardProps, backProp')
              )
            
algCell2 ::  Cell (Fix Cell,  (Deltas -> Deltas)) ->  (Fix Cell, (Deltas -> Deltas))
algCell2 InputCell 
    = (Fx InputCell, id)
algCell2 cell
    =   let (state, deltas, (nextCell, deltaTotalFunc)) = (_cellState cell, _cellDeltas cell, _innerCell cell)

            Deltas deltaW1 deltaU1 deltaB1 deltaXs1 = deltas
            deltaTotalFunc' = (\deltaTotal -> 
                let Deltas deltaW2 deltaU2 deltaB2 deltaXs2 = deltaTotal
                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
                    deltaB_total = (eleadd deltaB1 deltaB2)
                    deltaXs      = deltaXs1 ++ deltaXs2
                in Deltas deltaW_total deltaU_total deltaB_total deltaXs) . deltaTotalFunc

        in  (Fx (cell {_innerCell = nextCell}), deltaTotalFunc') --

compGates :: HyperParameters -> [Double] -> [Double] -> Gates
compGates (weightsW, weightsU, biases) x h 
    =   let p = (V.fromList [map sigmoid, map sigmoid, map tanh, map sigmoid])  :: V.Vec4 ([Double] -> [Double])
        in  V.zipWith ($) (p) (eleadd3v (V.map (mvmulk x) weightsW) (V.map (mvmulk h) weightsU) biases)

compDGates :: Gates -> [Double] -> [Double] -> [Double] -> [Double] -> [Double]
compDGates gate dOut dState state lastState 
    = let   d_f        = elemul4 dState (gate ! 1) lastState  (map sub1 (gate ! 1))
            d_i        = elemul4 dState (gate ! 2) (gate ! 3)  (map sub1 (gate ! 2))
            d_a        = elemul3 dState (gate ! 2) (map (sub1 . sqr) (gate ! 3))
            d_o        = elemul4 dOut   (gate ! 4) (map tanh state)  (map sub1 (gate ! 4))
      in    d_f ++ d_i ++ d_a ++ d_o

updateParameters ::  Layer k -> Deltas -> Layer k
updateParameters layer delta_total
    =   let Deltas deltaW_total deltaU_total deltaB_total deltaXs = delta_total
            (weights_w,weights_u,biases) = fromJust $ layer ^? hparams
            w = concat $ V.toList weights_w
            u = concat $ V.toList weights_u
            b = concat $ V.toList biases
            w'     = V.fromList $ map cons $ elesubm w (map2 (0.1 *) deltaW_total) 
                                            -- (elesubm (elesubm w (map2 (0.1 *) deltaW_total)) (replicate 4 (head deltaXs)))
           
            u'     = V.fromList $ map cons $ elesubm u (map2 (0.1 *) deltaU_total)
            b'     = V.fromList $ map cons $ elesub  b (map (0.1 *)  deltaB_total)

        in layer & hparams .~ (w', u', b')

emptyParams = ((V.fromList (replicate 4 [[]])),(V.fromList (replicate 4 [[]])),(V.fromList (replicate 4 [])))
emptyGates  = (V.fromList (replicate 4 []))

initForwardProp :: Int -> Int -> HyperParameters -> Inputs -> ForwardProp
initForwardProp h d params sample 

    = ForwardProp (V.fromList (replicate 4 [])) [] [] (replicate h 0.0) (replicate h 0.0) params sample

initBackProp :: Int -> Int -> Maybe [[Double]] -> Maybe Weights -> BackProp
initBackProp h d deltaX weights 
    = BackProp (replicate h 0) (replicate h 0) (replicate (h*d) 0) (replicate h 0) deltaX 

initDelta :: Int -> Int -> Deltas
initDelta h d = Deltas (fillMatrix (4 * h) (d) 0.0) (fillMatrix (4 * h) (h) 0.0) (replicate  (4 * h) 0.0) [[]]

example =   Fx (Layer (V.fromList [[[0.7]],  [[0.95]],  [[0.45]],   [[0.6]]],
                       V.fromList [[[0.2]]      ,  [[0.8]]      ,   [[0.15]]   ,    [[0.25]]],
                       V.fromList [[0.0]       , [0.0]        , [0.0]        ,    [0.0]])
                       (Fx (EndCell [0.0] NoDeltas
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas 
                                    (Fx (Cell [0] NoDeltas
                                        (Fx (Cell [0] NoDeltas
                                            (Fx InputCell)))))))))))
            (Fx (Layer (V.fromList [[[0.45]],  [[0.8]],  [[0.25]],   [[0.4]]],
                     V.fromList [[[0.6]]      ,  [[0.3]]      ,   [[0.3]]     ,    [[0.7]]],
                     V.fromList [[0.0]       , [0.0]         , [0.0]        ,    [0.0]])
                     (Fx (EndCell [0.0] NoDeltas
                        (Fx (Cell [0] NoDeltas 
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas  
                                    (Fx (Cell [0] NoDeltas 
                                        (Fx InputCell))))))))))) (Fx InputLayer))))
example' =   
            (Fx (Layer (V.fromList [[[0.7]],  [[0.95]],  [[0.45]],   [[0.6]]],
                     V.fromList [[[0.1]]      ,  [[0.8]]      ,   [[0.15]]     ,    [[0.25]]],
                     V.fromList [[0.15]       , [0.65]         , [0.2]        ,    [0.1]])
                    (Fx (EndCell [0.0] NoDeltas 
                        (Fx (Cell [0] NoDeltas  
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas 
                                    (Fx (Cell [0] NoDeltas 
                                        (Fx InputCell))))))))))) (Fx InputLayer)))

-- runRecurrent = print $ show $ runLayer example sample
--             where sample =  [([0.8,0.4],[0.5]),([0.5,0.1], [0.25])]
                         
runRecurrent' = print $ show $ runLayer example' sample
            where sample =  [([1, 2],[0.5]),([0.5, 3], [1.25])]
                         
runDNA :: [[([Double], [Double])]] -> IO ()
runDNA samples = do 
    print $ show $ trains example' samples

runCell :: Layer k -> Layer k 
runCell InputLayer = InputLayer
runCell (Layer params cells innerLayer)
    = let sample = [([1,2],[0.5]),([0.5,3], [1.25])]
          dDim = length . fst $ head sample
          hDim = 1-- let (w,u,b) = params in length $ w ! 1
   
          initialForwardProp = initForwardProp hDim dDim params sample
          initialBackProp    = initBackProp hDim dDim Nothing Nothing
          initialDeltaTotal  = initDelta  hDim dDim

          (cellf, deltaTotalFunc) = 
                                    let h =  (\(c, f) -> (c, f [initialForwardProp], initialBackProp))

                                    in  ((cata algCell2) . (meta algCell h coalgCell)) cells

          deltaTotal                = deltaTotalFunc initialDeltaTotal

      in  updateParameters (Layer params cellf innerLayer) deltaTotal
