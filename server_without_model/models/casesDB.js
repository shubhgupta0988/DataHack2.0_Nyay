const mongoose = require('mongoose')
const caseSchema = mongoose.Schema({
    caseID: {
        type: String,
        required: true
    },
    caseTitle: {
        type: String,
        required: true
    },
    caseDescription: {
        type: String,
        required: true
    },
    caseType: {
        type: String,
        required: true
    },
    caseStatus: {
        type: String,
        required: true
    },
    caseFilingDate: {
        type: Date,
        required: true
    },
    caseFilingCourt: {
        type: String,
        required: true
    },
    caseJudge: {
        type: String,
        required: true
    },
    caseOpponent: {
        type: String,
        required: true
    },
    caseNotes: {
        type: String,
        required: true
    },
    caseOutcome: {
        type: String,
        required: true
    },
    caseDates: {
        type: Array,
        default: [],
    },
    caseFiles: {
        type: Array,
        default: [],
    },
    clientDetails: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User"
    },
    lawyerDetails: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Lawyer"
    },
},
{
    collection: "cases"
})

module.exports = mongoose.model("Cases", caseSchema)