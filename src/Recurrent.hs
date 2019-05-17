
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
                        _des_out     :: [Double],
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
                        _nextLayerDXs       :: Maybe [[Double]],
                        _nextLayerDGates    :: [[Double]]
                    } deriving Show
makeLenses ''BackProp

data Deltas  = Deltas {
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double],
                        deltaXs          :: [[Double]],
                        deltaGates       :: [[Double]]
                }
                | NoDeltas

instance Show Deltas where 
    show (Deltas w u b x g) =
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
runLayer layer sample = let f = \(layer', forwardProp') -> (layer', forwardProp' sample, (initBackProp 1 2 Nothing []))
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
                    in  (layerFP:fps)) . nextForwardProp
      in (Fx (Layer params cells innerLayer), forwardProp) 

coalgLayer :: (Fix Layer, [[ForwardProp]], BackProp) -> Layer (Fix Layer, [[ForwardProp]], BackProp)
coalgLayer (Fx InputLayer, fp, bp)
    = InputLayer
coalgLayer (Fx (Layer params cells innerLayer), fps, backProp)
    =   let w                   = params^._1
            (hDim, dDim)        = (length $ w ! 1, length $ head $ w ! 1)
            (cell, deltaFunc)   = hylo algCell2 coalgCell (cells, head fps, backProp)
            deltaTotal          = deltaFunc (initDelta hDim dDim)

            backProp'           = initBackProp hDim dDim (Just $ deltaXs deltaTotal) (deltaGates deltaTotal) 
            showCost            = trace ((\z -> showFullPrecision $ read $ formatFloatN z 8) $ sum $ map sqr $ concat $ deltaW deltaTotal) 
            writeCost           = writeResult ((\z -> showFullPrecision $ read $ formatFloatN (z * 10000) 8) $ sum $ map (abs) $ concat $ deltaW deltaTotal) :: a -> a
 
        in  case innerLayer of (Fx (InputLayer)) ->   updateParameters (Layer params cell (innerLayer, tail fps, backProp')) deltaTotal
                               _             -> writeCost (updateParameters (Layer params cell (innerLayer, tail fps, backProp')) deltaTotal)

algCell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
algCell InputCell = 
    (Fx InputCell, \x -> x ++ x)
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

coalgCell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp) 
coalgCell (Fx InputCell, forwardProps, backProp)
    = InputCell
coalgCell (Fx cell, forwardProps, backProp)
  = let fp = head forwardProps
        lastState               = (head (tail forwardProps)) ^. state 
        (gate, updatedState)    = (fp ^. gates, fp ^. state)
        (weightsW, weightsU)    = mapT2 (V.foldr (++) [[]]) (fp^.params._1, fp^.params._2)

        BackProp dState_next deltaOut_next deltaGates_next f_next nextLayerDxs nextLayerDgates = backProp

        error = case nextLayerDxs of Nothing -> (elesub (fp^.output)  (fp^.des_out))
                                     Just nextLayerDxs' -> elesub (fp^.output) (map ((-1) * ) (head nextLayerDxs')) --(fp^.output)) trace (show nextLayerDgates) $ elemul (map (((head nextLayerDgates) !! 3) *) (head nextLayerDxs')) (fp ^. input)

        dOut =  
            case (cell) 
            of  (EndCell {})  -> error
                (Cell {})     -> eleadd deltaOut_next error

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
                                & nextLayerDGates .~ tail nextLayerDgates
                                & nextF      .~ (gate ! 1)
                                & nextLayerDXs .~   case backProp ^. nextLayerDXs 
                                                    of   Just dxs -> Just (tail dxs) 
                                                         Nothing  -> Nothing
    in  
        (cell & cellState .~ updatedState
              & cellDeltas .~ (Deltas deltaW deltaU deltaB [deltaX] [deltaGates])
              & innerCell .~ (fromJust (cell ^? innerCell), tail forwardProps, backProp')
              )
            
algCell2 ::  Cell (Fix Cell,  (Deltas -> Deltas)) ->  (Fix Cell, (Deltas -> Deltas))
algCell2 InputCell 
    = (Fx InputCell, id)
algCell2 cell
    =   let (state, deltas, (nextCell, deltaTotalFunc)) = (_cellState cell, _cellDeltas cell, _innerCell cell)

            Deltas deltaW1 deltaU1 deltaB1 deltaXs1 deltaGates1 = deltas
            deltaTotalFunc' = (\deltaTotal -> 
                let Deltas deltaW2 deltaU2 deltaB2 deltaXs2 deltaGates2 = deltaTotal
                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
                    deltaB_total = (eleadd deltaB1 deltaB2)
                    deltaXs      = deltaXs1 ++ deltaXs2
                    deltaGates   = deltaGates1 ++ deltaGates2
                in Deltas deltaW_total deltaU_total deltaB_total deltaXs deltaGates2) . deltaTotalFunc

        in  (Fx (cell {_innerCell = nextCell}), deltaTotalFunc') --

compGates :: HyperParameters -> [Double] -> [Double] -> Gates
compGates (weightsW, weightsU, biases) x h 
    =   let p =(V.fromList [map sigmoid, map sigmoid, map tanh, map sigmoid])  :: V.Vec4 ([Double] -> [Double])
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
    =   let Deltas deltaW_total deltaU_total deltaB_total deltaXs deltaGates = delta_total
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

initBackProp :: Int -> Int -> Maybe [[Double]] -> [[Double]] -> BackProp
initBackProp h d deltaX deltaGates 
    = BackProp (replicate h 0) (replicate h 0) (replicate (h*d) 0) (replicate h 0) deltaX deltaGates

initDelta :: Int -> Int -> Deltas
initDelta h d = Deltas (fillMatrix (4 * h) (d) 0.0) (fillMatrix (4 * h) (h) 0.0) (replicate  (4 * h) 0.0) [[]] [[]]


deep_lstm = do 
    weights_w_a <- randMat3D 2 1 4 
    weights_u_a <- randMat3D 1 1 4
    weights_w_b <- randMat3D 1 1 4
    weights_u_b <- randMat3D 1 1 4
    let biases_a = replicate 4 [0.0]
        biases_b = replicate 4 [0.0] 
    return $ Fx (Layer (V.fromList weights_w_b, V.fromList weights_u_b, V.fromList biases_b)
                    (Fx (EndCell [0] NoDeltas
                        (Fx (Cell [0] NoDeltas 
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas
                                    (Fx (Cell [0] NoDeltas
                                        (Fx InputCell)))))))))))
                (Fx (Layer (V.fromList weights_w_a, V.fromList weights_u_a, V.fromList biases_a)
                    (Fx (EndCell [0.0] NoDeltas
                        (Fx (Cell [0] NoDeltas 
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas  
                                    (Fx (Cell [0] NoDeltas 
                                        (Fx InputCell))))))))))) (Fx InputLayer))))
lstm =  do
    weights_w <- randMat3D 2 1 4
    weights_u <- randMat3D 1 1 4
    let biases = replicate 4 [0.0] 
    return  (Fx (Layer (V.fromList weights_w, V.fromList weights_u, V.fromList biases)
                (Fx (EndCell [0.0] NoDeltas 
                        (Fx (Cell [0] NoDeltas  
                            (Fx (Cell [0] NoDeltas 
                                (Fx (Cell [0] NoDeltas 
                                    (Fx (Cell [0] NoDeltas 
                                        (Fx InputCell))))))))))) 
                (Fx InputLayer)))

               
runRecurrent :: [[([Double], [Double])]] -> IO ()
runRecurrent samples = do 
    network <- deep_lstm
    print $ show $ trains network samples

runCell :: Layer k -> Layer k 
runCell InputLayer = InputLayer
runCell (Layer params cells innerLayer)
    = let sample = [([1,2],[0.5]),([0.5,3], [1.25])]
          dDim = length . fst $ head sample
          hDim = 1-- let (w,u,b) = params in length $ w ! 1
   
          initialForwardProp = initForwardProp hDim dDim params sample
          initialBackProp    = initBackProp hDim dDim Nothing []
          initialDeltaTotal  = initDelta  hDim dDim

          (cellf, deltaTotalFunc) = 
                                    let h =  (\(c, f) -> (c, f [initialForwardProp], initialBackProp))

                                    in  ((cata algCell2) . (meta algCell h coalgCell)) cells

          deltaTotal                = deltaTotalFunc initialDeltaTotal

      in  updateParameters (Layer params cellf innerLayer) deltaTotal
