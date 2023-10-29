import express from 'express';
import {createLawyer, getLawyer, updateLawyer, deleteLawyer, getRatings, getReviews} from '../controllers/lawyerController.js';

const router = express.Router();
router.route('/register').post(createLawyer);
router.route('/:id').get(getLawyer);
router.route('/:id').put(updateLawyer).delete(deleteLawyer);
router.route('/rate/:id').get(getRatings);
router.route('/review/:id').get(getReviews);

export default router;