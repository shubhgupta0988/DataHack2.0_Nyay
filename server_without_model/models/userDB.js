import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
    {
        fullName: {
            type: String,
            required: true,
            min: 3,
            max: 30,
        },
        email: {
            type: String,
            required: true,
            max: 50,
            unique: true,
        },
        phone: {
            type: String,
            required: true,
            max: 10,
            unique: true,
        },

        personalInfo: {
            gender: {
                type: String,
            },
            dob: {
                type: String,
            },
            password: {
                type: String,
                // required: true,
                min: 6,
            },
            profilePicture: {
                type: String,
                default: "",
            },
            city: {
                type: String,
                max: 50
            },
            state: {
                type: String,
                max: 50
            },
        },

        languages: {
            type: Array,
            default: [],
        },
        accountType: {
            type: String,
            default: "user",
        },

        ratings: [
            {
                type: mongoose.Schema.Types.ObjectId,
                ref: "Ratings",
            },
        ],
        reviews: [
            {
                type: mongoose.Schema.Types.ObjectId,
                ref: "Reviews",
            },
        ],
        cases: [
            {
                type: mongoose.Schema.Types.ObjectId,
                ref: "Cases",
            },
        ],
    },
    {
        timestamps: true
    },
    {
        collection: 'users'
    },
);

export const User = mongoose.model("User", userSchema);