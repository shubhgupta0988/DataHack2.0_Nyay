import mongoose from "mongoose";

const reviewSchema = mongoose.Schema(
    {
        users: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Uers",
        },
        review: {
            type: String,
            min: 2,
            max: 500,
            required: true
        },
        lawyer: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Lawyers",
        },
        userName: {
            type: String,
        },
        lawyerName: {
            type: String,
        },
    },
    {
        timestamps: true
    },
    {
        collection: "reviews"
    }
)

export const Reviews = mongoose.model("Reviews", reviewSchema)