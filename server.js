const express = require('express');
const app = express()
const cors = require('cors')
const bcrypt = require('bcrypt')
const mongoose = require('mongoose')

const PORT = 1337;

app.use(cors())
app.use(express.urlencoded({ extended: false}))
app.use(express.json());

mongoose.connect('mongodb://localhost:27017/get-a-mentor')

app.get('/', (req, res)=> {
    res.send('hello');
})


//Login
app.get('/login', (req, res) =>{
    res.send('Login Window')
})

app.post('/login', (req, res) =>{
    res.send('Login Window')
})


//Register
app.post('/api/register-mentee', async (req, res) =>{
    try{
        await Mentee.create({
            type: req.body.type,
            firstName: req.body.firstName,
            lastName: req.body.lastName,
            email: req.body.email,
            phone: req.body.phone,
            password: req.body.password
        })
        res.json({status:'ok'})
    }
    catch(err){
        console.log(err)
        res.json({status:'error'})
    }
    })

app.listen(PORT, () => console.log(`Express server currently running on port ${PORT}`));