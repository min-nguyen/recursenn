{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     StandaloneDeriving,
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

module Depend where

-- import Lib
-- import Data.Functor     
-- import Data.Foldable
-- import Data.Traversable
-- import Data.List
-- import Data.Ord
-- import Text.Show.Functions
-- import qualified Vector as V
-- import Vector (Vector((:-)))
-- import Debug.Trace

-- --------------------------------------------------------
-- -- DEPENDENT TYPED BUT INVALID FIX POINT RESTRICTIONS ON MATRIX SIZES
-- --------------------------------------------------------

-- type Weights            = [[Double]]
-- type Biases             = [Double]
-- type Inputs             = [Double]
-- type Outputs            = [Double]
-- type Activation         =  Double  ->  Double
-- type Activation'        =  Double  ->  Double
-- type LossFunction       = [Double] -> [Double] -> Double
-- type DesiredOutput      = [Double]
-- type FinalOutput        = [Double]
-- type Deltas             = [Double]

-- -- Fix f = Fx (f)

-- -- cata_dep :: Functor f => (f a -> a) -> Fu f -> a
-- cata_dep alg = alg . fmap (cata_dep alg) . unFu
-- ana_dep alg =  Fu . fmap (ana_dep alg) . alg
-- --     where fmap' = fmap' . fmap alg
-- -- ana_dep :: Functor f => (a -> f a) -> (a -> Fix f)
-- -- ana_dep coalg = Fx . fmap (ana_dep coalg) . coalg
-- newtype Fu f = Fu f
-- unFu (Fu f) = f

-- data Matrix m n k
-- data MatrixExpr m n k =
--       Leaf (Matrix m n k)
--     | forall p. Mult (MatrixExpr m p k) (MatrixExpr p n k)

-- data Layer m n k  =
--     forall p q. Layer (V.Vector (V.Vector Double n ) m) (V.Vector Double n) Activation (Layer' n p k)
--     | InputLayer 
-- deriving instance Functor (Layer m n)

-- data Layer' m n k =
--       forall p. Layer' (V.Vector (V.Vector Double n ) m) (V.Vector Double n) Activation k
--     | InputLayer' 
-- deriving instance Functor (Layer' m n)
    
-- forward' :: Fractional a => V.Vector (V.Vector a i) o -> V.Vector a i -> (a -> a) -> ([[a]] -> [[a]]) -> ([[a]] -> [[a]]) 
-- forward' weights biases activate k 
--     = (\inputs ->  ((map activate (zipWith (+) 
--             ((V.toList $ V.map 
--                 ((sum)  . (zipWithPadding (*) (head inputs)) . (V.toList)) weights)) (V.toList biases))):inputs) ) . k

-- backward' :: (V.SingRep o, V.SingRep i, Fractional a) => V.Vector (V.Vector a i) o -> V.Vector a i -> [a] -> [a] -> V.Vector (V.Vector a i) o 
-- backward' weights biases input final_output 
--     = let   list_weights = (V.toList $ ((V.map V.toList weights)) )
--             learning_rate = 1
--             desired_output = 3
--             error = (sum final_output) / (fromIntegral $ length final_output)
--             new_list_weights = transpose $ map (zipWith (+) (map (\xi -> learning_rate * xi * (desired_output - error)) input )) (transpose list_weights)
--       in    V.unsafeFromList' $ map V.unsafeFromList' $ new_list_weights

-- alg' :: Layer' m n (Fu (Layer' n p k), ([Inputs] -> [Inputs]) )
--         -> (Fu (Layer' m n (Fu (Layer' n p k) )), ([Inputs] -> [Inputs]))
-- alg' (Layer' weights biases activate (innerLayer, forwardPass) )   
--     =  (Fu (Layer' weights biases activate innerLayer) , (forward' weights biases activate forwardPass) )
-- alg' (InputLayer' )                     
--     =  (Fu InputLayer', id )

-- coalg' :: (V.SingRep m, V.SingRep n) => 
--        (Fu (Layer' m n (Fu (Layer' n p k) )), [Inputs]) 
--        -> (Layer' m n (Fu (Layer' n p k), [Inputs]) )
-- coalg' (Fu (Layer' weights biases activate innerLayer), (x:y:ys))
--     =  (Layer' (backward' weights biases y x) biases activate (innerLayer, (x:ys)))
-- coalg' (Fu InputLayer', output)      
--     =  InputLayer'

-- train' :: (V.SingRep m, V.SingRep n) => (Fu (Layer' m n k)) -> LossFunction -> Inputs -> Fu (Layer' m n q)
-- train' neuralnet lossfunction sample = ana_dep coalg' $ (nn,  diff_fun [sample])
--   where 
--     (nn, diff_fun) = cata_dep alg' neuralnet

-- example' =  (Fx ( Layer' ((3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--              (Fx ( Layer' ((3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--               (Fx   InputLayer' ) ) ) ) )


--------------------------------------------------------
-- ATTEMPT AT MULTIPARAMETER FIX TYPES
--------------------------------------------------------



-- data Fox o i j f = Fo o i (f (Fox j o V.Nat f))

-- unFox :: Functor f => Fox o i j f -> f (Fox j o V.Nat f)
-- unFox (Fo o i x) = x

-- cataf :: Functor f => (f a -> a) -> Fox o i j f  -> a
-- cataf alg = alg . fmap (cataf alg) . unFox

-- data Layer' (o::V.Nat) (i::V.Nat) k where
--     Layer' :: V.Vector (V.Vector Double i ) o  -> V.Vector Double i  -> Activation -> k -> Layer' o i k
--     InputLayer' :: Layer' o i k 
--     deriving Show
    
-- instance Functor (Layer' o i) where
--     fmap eval (Layer' weights biases activate k)      = Layer' weights biases activate (eval k) 
--     fmap eval (InputLayer' )                          = InputLayer' 


-- forward' :: Fractional a => V.Vector (V.Vector a i) o -> V.Vector a i -> (a -> a) -> ([[a]] -> [[a]]) -> ([[a]] -> [[a]]) 
-- forward' weights biases activate k 
--     = (\inputs ->  ((map activate (zipWith (+) 
--             ((V.toList $ V.map 
--                 ((sum)  . (zipWithPadding (*) (head inputs)) . (V.toList)) weights)) (V.toList biases))):inputs) ) . k

-- backward' :: (V.SingRep o, V.SingRep i, Fractional a) => V.Vector (V.Vector a i) o -> V.Vector a i -> [a] -> [a] -> V.Vector (V.Vector a i) o 
-- backward' weights biases input final_output 
--     = let   list_weights = (V.toList $ ((V.map V.toList weights)) )
--             learning_rate = 1
--             desired_output = 3
--             error = (sum final_output) / (fromIntegral $ length final_output)
--             new_list_weights = transpose $ map (zipWith (+) (map (\xi -> learning_rate * xi * (desired_output - error)) input )) (transpose list_weights)
--       in    V.unsafeFromList' $ map V.unsafeFromList' $ new_list_weights

-- alg' :: Layer' o i (Fix (Layer' o i) , ([Inputs] -> [Inputs]) ) 
--         -> (Fix (Layer' o i) , ([Inputs] -> [Inputs]))
-- alg' (Layer' weights biases activate (innerLayer, forwardPass) )   
--     =  (Fx (Layer' weights biases activate innerLayer ) , (forward' weights biases activate forwardPass) )
-- alg' (InputLayer' )                     
--     =  (Fx InputLayer', id )

-- coalg' :: (V.SingRep o, V.SingRep i) => (Fix (Layer' o i), [Inputs]) -> Layer' o i (Fix (Layer' o i), [Inputs]) 
-- coalg' (Fx (Layer' weights biases activate innerLayer), (x:y:ys))
--     =  Layer' (backward' weights biases y x) biases activate (innerLayer, (x:ys))
-- coalg' (Fx InputLayer', output)      
--     =  InputLayer' 

-- train' :: (V.SingRep o, V.SingRep i) => Fix (Layer' o i) -> LossFunction -> Inputs -> Fix (Layer' o i)
-- train' neuralnet lossfunction sample = ana coalg' $ (nn,  diff_fun [sample])
--   where 
--     (nn, diff_fun) = cata alg' neuralnet

-- example' =  (Fx ( Layer' ((3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-(3.0:-10.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--              (Fx ( Layer' ((3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-(3.0:-1.0:-2.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
--               (Fx   InputLayer' ) ) ) ) )



