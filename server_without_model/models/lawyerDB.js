import mongoose from "mongoose";

const lawyerSchema = new mongoose.Schema(
  {
    fullName: {
      type: String,
      // required: true,
      min: 3,
      max: 30,
    },
    email: {
      type: String,
      // required: true,
      max: 50,
      unique: true,
    },
    phone: {
      type: String,
      // required: true,
      max: 10,
      unique: true,
    },

    personalInfo: {
      officePhone: {
        type: String,
        max: 50,
        unique: true,
      },
      bio: {
        type: String,
        max: 500,
      },
      gender: {
        type: String,
        // required: true,
      },
      dob: {
        type: Date,
        // required: true,
      },
      password: {
        type: String,
        // required: true,
        min: 6,
        max: 8,
      },
      profilePicture: {
        type: String,
        default: "",
      },
      connections: {
        type: Array,
        default: [],
      },
      city: {
        type: String,
        max: 20,
        // required: true,
      },
      state: {
        type: String,
        max: 20,
        // required: true,
      },
      officeAdd: {
        type: String,
        max: 100,
      },
    },

    proffesionalInfo: {
      notableCases: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Cases",
      },
      barAssociation: {
        type: String,
        max: 50,
      },
      licenseNumber: {
        type: String,
        max: 50
      },
      barYear: {
        type: Number,
        default: 0
      },
      barState: {
        type: String,
        max: 50
      },
      practiceAreas: {
        type: Array,
        default: []
      },
      experience: {
        type: Number,
        default: 0
      },
      education: [
        {
          year: {
            type: Number,
            // required: true,
          },
          institute: {
            type: String,
            // required: true,
          },
          result: {
            type: String,
            // required: true,
          },
          degree: {
            type: String,
            // required: true,
          },
        },
      ],
      specializations: {
        type: Array,
        default: [],
      },
      publications: {
        type: Array,
        default: [],
      },
      awards: {
        type: Array,
        default: [],
      },
    },

    languages: {
      type: Array,
      default: [],
    },
    fees: {
      type: Number,
      default: 0,
    },
    socialMedia: {
      type: Array,
      default: [],
    },
    accountType: {
      type: String,
      default: "lawyer",
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
    timestamps: true,
  },
  {
    collection: 'lawyers'
  },
);

export const Lawyers = mongoose.model("Lawyers", lawyerSchema);

