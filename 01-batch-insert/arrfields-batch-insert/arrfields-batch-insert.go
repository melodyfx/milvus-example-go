package main

import (
	"context"
	"fmt"
	"github.com/google/uuid"
	"log"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr     = `192.168.230.71:19530`
	nEntities, dim = 100, 5
	collectionName = "hello_array"

	msgFmt                         = "==== %s ====\n"
	idCol, randomCol, embeddingCol = "ID", "random", "embeddings"
)

func convertToStringBytes3D(input [][]string) [][][]byte {
	// 初始化输出切片
	output := make([][][]byte, len(input))
	for i, row := range input {
		// 为每一行初始化二维字节切片
		output[i] = make([][]byte, len(row))
		for j, str := range row {
			// 将每个字符串转换为字节切片，并存储
			output[i][j] = []byte(str)
		}
	}
	return output
}

func main() {
	ctx := context.Background()

	log.Printf(msgFmt, "start connecting to Milvus")
	c, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus, err: ", err.Error())
	}
	defer c.Close()

	// delete collection if exists
	//has, err := c.HasCollection(ctx, collectionName)
	//if err != nil {
	//	log.Fatalf("failed to check collection exists, err: %v", err)
	//}
	//if has {
	//	c.DropCollection(ctx, collectionName)
	//}

	// create collection
	log.Printf(msgFmt, fmt.Sprintf("create collection, `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription("hello_array is the simplest demo to introduce the APIs").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName(randomCol).WithDataType(entity.FieldTypeArray).WithElementType(entity.FieldTypeVarChar).WithMaxLength(1000).WithMaxCapacity(10)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil { // use default shard number
		log.Fatalf("create collection failed, err: %v", err)
	}

	// insert data
	log.Printf(msgFmt, "start inserting random entities")
	idList, randomList := make([]int64, 0, nEntities), make([][]string, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	rand.Seed(time.Now().UnixNano())

	// generate data
	start := 0
	end := 100
	for i := start; i < end; i++ {
		idList = append(idList, int64(i))
	}
	for i := start; i < end; i++ {
		data := make([]string, 0, 10)
		for j := 0; j < 10; j++ {
			u, _ := uuid.NewUUID()
			data = append(data, u.String())
		}
		randomList = append(randomList, data)
	}
	for i := start; i < end; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}
		embeddingList = append(embeddingList, vec)
	}
	idColData := entity.NewColumnInt64(idCol, idList)
	randomColData := entity.NewColumnVarCharArray(randomCol, convertToStringBytes3D(randomList))
	embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)

	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `hello_array, err: %v", err)
	}

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}

	// build index
	//log.Printf(msgFmt, "start creating index HNSW")
	//idx, err := entity.NewIndexHNSW(entity.COSINE, 8, 64)
	//if err != nil {
	//	log.Fatalf("failed to create ivf flat index, err: %v", err)
	//}
	//
	//if err := c.CreateIndex(ctx,
	//	collectionName,
	//	embeddingCol,
	//	idx,
	//	false,
	//	client.WithIndexName("idx_"+embeddingCol)); err != nil {
	//	log.Fatalf("failed to create index, err: %v", err)
	//}
	//
	//log.Printf(msgFmt, "start loading collection")
	//err = c.LoadCollection(ctx, collectionName, false)
	//if err != nil {
	//	log.Fatalf("failed to load collection, err: %v", err)
	//}
}
