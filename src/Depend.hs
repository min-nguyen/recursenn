module Depend where

-- data Layer' (o::Nat) (i::Nat) k where
--     Layer' :: Vector o (Vector i Double) -> Vector i Double -> Activation -> k -> Layer' o i k
--     InputLayer' :: Layer' o i k 
--     -- deriving Show

-- instance Functor (Layer' o i) where
--     fmap eval (Layer' weights biases activate k)      = Layer' weights biases activate (eval k) 
--     fmap eval (InputLayer' )                          = InputLayer' 

-- forward' :: Fractional a => Vector o (Vector i a) -> Vector i a -> (a -> a) -> (Vector n [a] -> Vector ('Succ n) [a]) -> (Vector n [a] -> Vector ('Succ ('Succ n)) [a]) 
-- forward' weights biases activate k 
--     = (\inputs -> Vcons (map activate (zipWith (+) 
--             ((vtoList $ vmap ((sum)  . (zipWithPadding (*) (vhead inputs)) . (vtoList)) weights)) (vtoList biases))) inputs) . k
  
-- backward' :: Fractional a => Vector o (Vector i a) -> Vector i a -> Vector i a -> Vector r a -> [[a]]
-- backward' weights biases input final_output 
--     = vtranspose $ vmap (vzipWith (+) (vmap (\xi -> learning_rate * xi * (desired_output - error)) input )) (vtranspose weights)
--         where   learning_rate = 1
--                 desired_output = 3
--                 error = (sum final_output) / (fromIntegral $ length final_output)

-- alg' :: Layer' o i (Fix (Layer' j o), (Vector n Inputs -> Vector ('Succ n) Inputs) ) 
--         -> (Fix (Layer' j o), (Vector n Inputs -> Vector ('Succ ('Succ n)) Inputs))
-- alg' (Layer' weights biases activate (innerLayer, forwardPass) )   
--     =  (Fx (Layer' weights biases activate innerLayer ) , (forward' weights biases activate forwardPass) )
-- alg' (InputLayer' )                     
--     =  (Fx InputLayer', ((vsnoc []) . (vsnoc [])) ) -- issue with id, can't return a function of Vector n Inputs -> Vector ('Succ ('Succ n)) Inputs
