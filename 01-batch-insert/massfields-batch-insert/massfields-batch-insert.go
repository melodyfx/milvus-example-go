package main

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"log"
	"math/rand"
	"strconv"
)

const (
	milvusAddr     = `192.168.0.107:19530`
	dim            = 3
	collectionName = "hello_iterator"

	msgFmt                                     = "==== %s ====\n"
	idCol, randomCol, addressCol, embeddingCol = "ID", "random", "address", "embeddings"
	boolCol, int8Col, int16Col, int32Col       = "boolCol", "int8Col", "int16Col", "int32Col"
	totalRows                                  = 500
	batchSize                                  = 100
)

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
	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("failed to check collection exists, err: %v", err)
	}
	if has {
		//c.DropCollection(ctx, collectionName)
		log.Println("collection exists")
		return
	}

	// create collection
	log.Printf(msgFmt, fmt.Sprintf("create collection, `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription("hello_milvus is the simplest demo to introduce the APIs").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName(randomCol).WithDataType(entity.FieldTypeDouble)).
		WithField(entity.NewField().WithName(addressCol).WithDataType(entity.FieldTypeVarChar).WithTypeParams(entity.TypeParamMaxLength, "50")).
		WithField(entity.NewField().WithName(boolCol).WithDataType(entity.FieldTypeBool)).
		WithField(entity.NewField().WithName(int8Col).WithDataType(entity.FieldTypeInt8)).
		WithField(entity.NewField().WithName(int16Col).WithDataType(entity.FieldTypeInt16)).
		WithField(entity.NewField().WithName(int32Col).WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, 1); err != nil {
		log.Fatalf("create collection failed, err: %v", err)
	}
	// 分批数
	groupNum := totalRows / batchSize
	if totalRows%batchSize > 0 {
		groupNum++ // 如果有余数，则需要多写一批
	}
	// 分批写入数据
	for i := 0; i < groupNum; i++ {
		// 计算当前批次的起始和结束索引
		start := i * batchSize
		end := start + batchSize
		if end > totalRows {
			end = totalRows // 确保不会超出总数
		}
		// insert data
		idList := make([]int64, 0, batchSize)
		randomList := make([]float64, 0, batchSize)
		addressList := make([]string, 0, batchSize)
		boolList := make([]bool, 0, batchSize)
		int8List := make([]int8, 0, batchSize)
		int16List := make([]int16, 0, batchSize)
		int32List := make([]int32, 0, batchSize)
		embeddingList := make([][]float32, 0, batchSize)

		// generate data
		for i := start; i < end; i++ {
			idList = append(idList, int64(i))
		}
		for i := start; i < end; i++ {
			randomList = append(randomList, rand.Float64())
		}
		for i := start; i < end; i++ {
			addressList = append(addressList, "wuhan"+strconv.Itoa(i))
		}
		for i := start; i < end; i++ {
			boolList = append(boolList, true)
		}
		for i := start; i < end; i++ {
			int8List = append(int8List, int8(i%125))
		}
		for i := start; i < end; i++ {
			int16List = append(int16List, int16(i))
		}
		for i := start; i < end; i++ {
			int32List = append(int32List, int32(i))
		}
		for i := start; i < end; i++ {
			vec := make([]float32, 0, dim)
			for j := 0; j < dim; j++ {
				vec = append(vec, rand.Float32())
			}
			embeddingList = append(embeddingList, vec)
		}
		idColData := entity.NewColumnInt64(idCol, idList)
		randomColData := entity.NewColumnDouble(randomCol, randomList)
		addressColData := entity.NewColumnVarChar(addressCol, addressList)
		boolColData := entity.NewColumnBool(boolCol, boolList)
		int8ColData := entity.NewColumnInt8(int8Col, int8List)
		int16ColData := entity.NewColumnInt16(int16Col, int16List)
		int32ColData := entity.NewColumnInt32(int32Col, int32List)
		embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)

		if _, err := c.Insert(ctx, collectionName, "",
			idColData,
			randomColData,
			addressColData,
			boolColData,
			int8ColData,
			int16ColData,
			int32ColData,
			embeddingColData); err != nil {
			log.Fatalf("failed to insert random data into `hello_milvus, err: %v", err)
		}
		log.Printf("inserted:%d\n", end)
	}

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}

	log.Printf(msgFmt, "start creating index HNSW")
	idx, err := entity.NewIndexHNSW(entity.COSINE, 15, 50)
	if err != nil {
		log.Fatalf("failed to create ivf flat index, err: %v", err)
	}
	// build index
	if err := c.CreateIndex(ctx,
		collectionName,
		embeddingCol,
		idx,
		false,
		client.WithIndexName("idx_"+embeddingCol)); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	log.Printf(msgFmt, "start loading collection")
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}
}
