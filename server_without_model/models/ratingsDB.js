import mongoose from "mongoose";

const ratingSchema = mongoose.Schema(
    {
        user: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Users",
        },
        rating: {
            type: Number,
            required: true,
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
        collection: "ratings"
    }
)

export const Ratings = mongoose.model("Ratings", ratingSchema);