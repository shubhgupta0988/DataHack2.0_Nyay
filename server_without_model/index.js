import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import connectDB from "./config/dbSetup.js";
import userRoute from "./routes/userRoutes.js";
import lawyerRoute from "./routes/lawyerRoutes.js";

const app = express();
dotenv.config();
app.use(cors());
const PORT = process.env.PORT || 5000;
connectDB();

app.use(express.urlencoded({ extended: true }));
app.listen(PORT, () => {
  console.log(`Server Started On http://localhost:${PORT}`);
});
app.use(express.json({ limit: "5mb" }));

app.use("/user/", userRoute);
app.use("/lawyer/", lawyerRoute);