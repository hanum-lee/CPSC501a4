
array_of_ID

desiredBookingID

array_length

midpoint = array_length/2
1,2,3,4

function binarySearch(int starting, int last){
    midpoint = floor((last - starting) / 2 + starting)
    if(midpoint < 1 || midpoint > array_length-1)
    if(array_of_ID[midpoint] < desiredBookingID){
        binarySearch(starting, last/2)
    }else if (array_of_ID[midpoint] > desiredBookingID){
        binarySearch(midpoint, last)
    }else{
        return index
    }
}

