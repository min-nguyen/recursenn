{-# LANGUAGE
  
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     GADTs,
     DataKinds,
     KindSignatures,
     TemplateHaskell #-}

module Vector where

import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions

-- class SingRep n where
--     sing :: Nat n
  
-- instance SingRep ('Zero) where
--     sing = Zero

-- instance SingRep n => SingRep ('Succ n) where
--     sing = Succ (sing :: Nat n)

-- data SingInstance (n :: Nat) where
--     SingInstance :: SingRep n => SingInstance n

-- singInstance :: Nat n -> SingInstance n
-- singInstance Vnil     = SingInstance
-- singInstance (Succ n) =
--     case singInstance n of
--         SingInstance -> SingInstance
  

data Nat = Zero | Succ Nat deriving Show

data Vector (n :: Nat) (a :: *) where
    Vnil :: Vector 'Zero a
    Vcons :: a -> Vector n a -> Vector ('Succ n) a
    -- deriving Show

type Matrix (o :: Nat) (i :: Nat) a = Vector o (Vector i a) 

-- toVector :: [a] -> Vector n a
-- toVector [] = Vnil 
-- toVector (x:xs) = Vcons x (toVector xs)

vtoList :: Vector n a -> [a]
vtoList (Vcons a v) = (a:(vtoList v))
vtoList Vnil = []

mtoList :: Matrix n m a -> [[a]]
mtoList m = vtoList (vmap vtoList m)

vhead :: Vector n a -> a
vhead (Vcons a v) = a

vtail :: Vector ('Succ n) a -> Vector n a
vtail (Vcons a v) = v 

vmap :: (a -> b) -> Vector n a -> Vector n b
vmap f (Vcons a v) = Vcons (f a) (vmap f v)
vmap f Vnil = Vnil

vzipWith :: (a -> b -> c) -> Vector n a -> Vector n b -> Vector n c
vzipWith f (Vcons a v0) (Vcons b v1) = Vcons (f a b) (vzipWith f v0 v1)
vzipWith f (Vnil) (Vnil) = Vnil

vsum :: Num a => Vector n a -> a
vsum (Vcons a v) = a + (vsum v)
vsum Vnil = 0

-- vtranspose :: Matrix n m a -> Matrix m n a
-- vtranspose Vnil = replicate' Vnil

