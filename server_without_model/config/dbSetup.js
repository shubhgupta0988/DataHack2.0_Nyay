import mongoose from 'mongoose';

const connectDB = async() => {
    mongoose.connect(process.env.MONGO_URL,{
        useNewUrlParser : true,
    }).then(console.log("Connected To MongoDB"))
    .catch(err => {console.error(err)})
}

export default connectDB;

// import mongoose from 'mongoose';
// const connectDB = () => {
//   mongoose.connect(process.env.MONGO_URL, { useNewUrlParser: true, useUnifiedTopology: true })
//     .then(() => {
//       console.log('Connected to MongoDB');
//     })
//     .catch(err => {
//       console.error('Failed to connect to MongoDB', err);
//       // Retry the connection after a delay
//       setTimeout(connectDB, 5000); // Retry after 5 seconds
//     });
// };

// connectDB(); // Initial connection attempt
// export default connectDB; 
