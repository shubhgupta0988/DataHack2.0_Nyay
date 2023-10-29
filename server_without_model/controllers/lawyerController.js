import asyncHandler from 'express-async-handler';
import {Lawyers} from '../models/lawyerDB.js';
import {Ratings} from '../models/ratingsDB.js';
import {Reviews} from '../models/reviewsDB.js';
import mongoose from "mongoose";
// const bcrypt = require('bcrypt');

const createLawyer = asyncHandler ( async (req, res) => {
    try{
        const {fullName, email, phone} = req.body;
        const lawyer = await Lawyer.create({
            fullName: fullName,
            email: email,
            phone: phone,
        })
        // const salt = await bcrypt.genSalt(10);
        // const hashedPassword = await bcrypt.hash(req.body.password, salt);
        res.status(200).json(lawyer);
    } catch (err) {
        console.log(err);
        res.status(500).json(err);
    }
});

const getLawyer = asyncHandler (async (req, res) => {
    try {
        const lawyer = await Lawyer.findOne({_id: req.params.id});
        const {password, updatedAt, ...other} = lawyer._doc;
        res.status(200).json(other);
    } catch(err) {
        res.status(404).json(err);
    }
})

const updateLawyer = asyncHandler (async (req, res) => {
    if(req.body.userId == req.params.id){
        try {
            const lawyer = await Lawyer.findByIdAndUpdate(req.params.id, { $set: req.body });
            res.status(200).json("Update successfully!");
        } catch(err) {
            res.status(500).json(err);
        }
    }
})

const deleteLawyer = asyncHandler (async (req, res) => {
    if(req.body.userId == req.params.id){
        try {
            const lawyer = await Lawyer.findOneAndDelete(req.params.id);
            res.status(200).json("Account has been deleted.");
        } catch(err) {
            res.status(500).json(err);
        }
    }
    else{
        return res.status(403).json("You can delete only your account.");
    }
})

const getRatings = asyncHandler (async (req, res) => {
    try {
        const lawyer = await Lawyers.findById(req.params.id);

        const ratings = await Promise.all(
            lawyer.ratings.map(rateId => {
                return Ratings.findById(rateId);
            }
        ))

        let rateList = [];
        let totalRating = 0;

        ratings.map((rates) => {
            const { _id, lawyerName, clientName, rating } = rates; // Use a different variable name to avoid conflict
            rateList.push({ rating });
            totalRating += rating;
        })

        const averageRating = totalRating / rateList.length;

        res.status(200).json({ averageRating });
    } catch(err) {
        res.status(404).json(err);
    }
})

const getReviews = asyncHandler (async (req, res) => {
    try {
        const lawyer = await Lawyers.findById(req.params.id);

        const reviews = await Promise.all(
            lawyer.reviews.map(revId => {
                return Reviews.findById(revId);
            }
        ))

        let revList = [];
        reviews.map((revs) => {
            const { _id, lawyerName, clientName, review } = revs;
            revList.push({ lawyerName, clientName, review });
        })

        res.status(200).json(revList);
    } catch(err) {
        res.status(404).json(err);
    }
})

export {createLawyer, getLawyer, updateLawyer, deleteLawyer, getRatings, getReviews};