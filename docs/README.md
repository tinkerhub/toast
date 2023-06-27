# API doc

## Get user id

Get the user id given an image

### URL

`/image_to_user`

### METHOD

POST

### PARAMETERS

Parameter | Description
--- | ---
`image` | **Required** Image


### SUCCESS RESPONSE

```
{
    "user_id": "string",
}
```

## Create user data

Create image user profile

### URL

`/user`

### METHOD

POST

### PARAMETERS

Parameter | Description
--- | ---
`image` | **Required** Image
`user_id` | **Required** TinkerHub Id

## RESPONSE

```
{
  "user_id": "string"
}
```

## Delete user profile

### URL

`/user/{user_id}`

### METHOD

DELETE

### PARAMETERS

Parameter | Description
--- | ---
`user_id` | **Required** TinkerHub Id

## RESPONSE

```
{
  "user_id": "string"
}
```







